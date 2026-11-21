// api: servidor HTTP del proyecto (también es el coordinador del cluster de nodos ML).
//
// Endpoints:
//   GET  /health                    estado del servicio
//   POST /api/predecir              probabilidad de anulación de una orden (responde en pocos ms)
//   POST /api/entrenar              entrena el modelo repartiendo el cálculo entre los nodos
//   GET  /api/entrenar/estado       progreso del entrenamiento en curso
//   GET  /api/entrenamientos        historial de entrenamientos (MongoDB)
//   GET  /api/nodos                 estado de los nodos del cluster
//   GET  /api/predicciones          últimas predicciones guardadas (MongoDB)
//   GET  /api/metricas              latencia de las predicciones y último entrenamiento
//   GET  /swagger/                  documentación interactiva (Swagger UI)
package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"compras/cluster/ml"
)

//go:embed openapi.yaml
var openapiYAML []byte

func entorno(clave, defecto string) string {
	if v := os.Getenv(clave); v != "" {
		return v
	}
	return defecto
}

// ---------------------------------------------------------------------------
// El modelo entrenado (pesos + datos para estandarizar)
// ---------------------------------------------------------------------------

type Modelo struct {
	W       []float64 // NF pesos y el sesgo al final
	Media   []float64 // media de las variables continuas
	Desv    []float64 // desviación de las variables continuas
	PesoPos float64   // peso que se usó para las órdenes anuladas al entrenar
}

var (
	modelo        Modelo
	hayModelo     bool
	candadoModelo sync.RWMutex // muchas consultas pueden leer a la vez; solo escribe quien entrena
)

func establecerModelo(m Modelo) {
	candadoModelo.Lock()
	defer candadoModelo.Unlock()
	modelo, hayModelo = m, true
}

func modeloActual() (Modelo, bool) {
	candadoModelo.RLock()
	defer candadoModelo.RUnlock()
	return modelo, hayModelo
}

// Lee modelo.txt (el archivo que genera ml_engine.go)
func cargarModelo(ruta string) (Modelo, error) {
	m := Modelo{PesoPos: 1}
	datos, err := os.ReadFile(ruta)
	if err != nil {
		return m, err
	}
	valores := map[string]float64{}
	for _, linea := range strings.Split(string(datos), "\n") {
		partes := strings.SplitN(strings.TrimSpace(linea), "=", 2)
		if len(partes) != 2 {
			continue
		}
		if v, err := strconv.ParseFloat(partes[1], 64); err == nil {
			valores[partes[0]] = v
		}
	}
	for j := 0; j <= ml.NF; j++ {
		v, ok := valores[fmt.Sprintf("w%d", j)]
		if !ok {
			return m, fmt.Errorf("falta w%d en %s (¿el modelo se entrenó con %d variables?)", j, ruta, ml.NF)
		}
		m.W = append(m.W, v)
	}
	for _, j := range ml.ColsContinuas {
		media, ok1 := valores[fmt.Sprintf("media_x%d", j)]
		desv, ok2 := valores[fmt.Sprintf("desv_x%d", j)]
		if !ok1 || !ok2 {
			return m, fmt.Errorf("falta la media o desviación de x%d: vuelve a ejecutar ml_engine.go", j)
		}
		m.Media = append(m.Media, media)
		m.Desv = append(m.Desv, desv)
	}
	if pp, ok := valores["peso_pos"]; ok && pp > 0 {
		m.PesoPos = pp
	}
	return m, nil
}

// ---------------------------------------------------------------------------
// Predicción
// ---------------------------------------------------------------------------

// Datos de una orden que se quiere evaluar
type EntradaPrediccion struct {
	Monto                float64 `json:"monto" bson:"monto"`
	LargoDescripcion     int     `json:"largo_descripcion" bson:"largo_descripcion"`
	FinDeMes             bool    `json:"fin_de_mes" bson:"fin_de_mes"`
	PersonaJuridica      bool    `json:"persona_juridica" bson:"persona_juridica"`
	Objeto               string  `json:"objeto" bson:"objeto"`
	TipoContratacion     string  `json:"tipo_contratacion" bson:"tipo_contratacion"`
	Sector               string  `json:"sector" bson:"sector"`
	TasaHistoricaEntidad float64 `json:"tasa_historica_entidad" bson:"tasa_historica_entidad"`
}

type RespuestaPrediccion struct {
	Probabilidad float64 `json:"probabilidad"`
	NivelRiesgo  string  `json:"nivel_riesgo"`
	DesdeCache   bool    `json:"desde_cache"`
	TiempoMs     float64 `json:"tiempo_ms"`
}

// El orden de estas listas es el mismo orden de las variables one-hot de data_loader.go
var objetos = []string{"BIENES", "SERVICIOS", "CONSULTORIA_OBRAS"}
var tiposContratacion = []string{"HASTA_8_UIT", "SIN_PROCESO", "PROCESO_SELECCION", "EXONERACION", "CATALOGO",
	"REGIMEN_ESPECIAL", "CONVENIO", "SERVICIO_PUBLICO", "DECRETO_URGENCIA"}
var sectores = []string{"SALUD", "EDUCACION", "MUNICIPAL", "GOBIERNO_REGIONAL", "SEGURIDAD_JUSTICIA", "SANEAMIENTO", "OTRO"}

func indice(lista []string, valor string) int {
	for i, v := range lista {
		if v == valor {
			return i
		}
	}
	return -1
}

func validar(e EntradaPrediccion) string {
	if e.Monto <= 0 {
		return "monto debe ser mayor que 0"
	}
	if e.LargoDescripcion < 0 {
		return "largo_descripcion no puede ser negativo"
	}
	if e.TasaHistoricaEntidad < 0 || e.TasaHistoricaEntidad > 1 {
		return "tasa_historica_entidad debe estar entre 0 y 1"
	}
	if indice(objetos, e.Objeto) < 0 {
		return "objeto debe ser uno de: " + strings.Join(objetos, ", ")
	}
	if indice(tiposContratacion, e.TipoContratacion) < 0 {
		return "tipo_contratacion debe ser uno de: " + strings.Join(tiposContratacion, ", ")
	}
	if indice(sectores, e.Sector) < 0 {
		return "sector debe ser uno de: " + strings.Join(sectores, ", ")
	}
	return ""
}

// Arma el vector de 24 variables igual que data_loader.go y calcula la probabilidad
func calcular(e EntradaPrediccion, m Modelo) float64 {
	var x [ml.NF]float64
	x[0] = math.Log1p(e.Monto)
	x[1] = float64(e.LargoDescripcion)
	if e.FinDeMes {
		x[2] = 1
	}
	if e.PersonaJuridica {
		x[3] = 1
	}
	x[4+indice(objetos, e.Objeto)] = 1
	x[7+indice(tiposContratacion, e.TipoContratacion)] = 1
	x[16+indice(sectores, e.Sector)] = 1
	x[23] = e.TasaHistoricaEntidad

	ml.Estandarizar(&x, m.Media, m.Desv)
	z := ml.Puntaje(m.W, &x)
	// Al entrenar, las anuladas pesaron "PesoPos" veces más. Restamos ln(PesoPos) para
	// que el resultado sea una probabilidad realista y no una inflada.
	return ml.Sigmoide(z - math.Log(m.PesoPos))
}

// Los umbrales salen de lo observado en validación: el 5 % de mayor riesgo tenía alrededor de 4 % de anuladas
func nivel(p float64) string {
	if p >= 0.04 {
		return "ALTO"
	}
	if p >= 0.02 {
		return "MEDIO"
	}
	return "BAJO"
}

// ---------------------------------------------------------------------------
// Latencias de las predicciones (para demostrar que se responde en menos de 100 ms)
// ---------------------------------------------------------------------------

var (
	latencias    []float64
	candadoLatencias sync.Mutex
)

func registrarLatencia(ms float64) {
	candadoLatencias.Lock()
	defer candadoLatencias.Unlock()
	latencias = append(latencias, ms)
	if len(latencias) > 1000 { // guardamos solo las últimas 1000
		latencias = latencias[len(latencias)-1000:]
	}
}

func resumenLatencias() map[string]interface{} {
	candadoLatencias.Lock()
	copia := append([]float64(nil), latencias...)
	candadoLatencias.Unlock()
	if len(copia) == 0 {
		return map[string]interface{}{"muestras": 0}
	}
	sort.Float64s(copia)
	suma := 0.0
	for _, v := range copia {
		suma += v
	}
	n := len(copia)
	return map[string]interface{}{
		"muestras":    n,
		"promedio_ms": suma / float64(n),
		"p50_ms":      copia[n*50/100],
		"p95_ms":      copia[n*95/100],
		"max_ms":      copia[n-1],
	}
}

// ---------------------------------------------------------------------------
// Handlers HTTP
// ---------------------------------------------------------------------------

func responderJSON(w http.ResponseWriter, codigo int, valor interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(codigo)
	json.NewEncoder(w).Encode(valor)
}

func responderError(w http.ResponseWriter, codigo int, mensaje string) {
	responderJSON(w, codigo, map[string]string{"error": mensaje})
}

func health(w http.ResponseWriter, r *http.Request) {
	_, ok := modeloActual()
	responderJSON(w, 200, map[string]interface{}{"estado": "ok", "modelo_cargado": ok, "hora": time.Now()})
}

func predecir(w http.ResponseWriter, r *http.Request) {
	inicio := time.Now()
	if r.Method != http.MethodPost {
		responderError(w, 405, "usa el método POST")
		return
	}
	var e EntradaPrediccion
	if err := json.NewDecoder(r.Body).Decode(&e); err != nil {
		responderError(w, 400, "el cuerpo no es un JSON válido")
		return
	}
	if msg := validar(e); msg != "" {
		responderError(w, 400, msg)
		return
	}
	m, ok := modeloActual()
	if !ok {
		responderError(w, 503, "todavía no hay un modelo cargado; entrena uno con POST /api/entrenar")
		return
	}

	// 1) buscamos en la caché (Redis); 2) si no está, calculamos y la guardamos
	clave := fmt.Sprintf("pred:%+v", e)
	prob, desdeCache := 0.0, false
	if texto, hay := cacheGet(clave); hay {
		if v, err := strconv.ParseFloat(texto, 64); err == nil {
			prob, desdeCache = v, true
		}
	}
	if !desdeCache {
		prob = calcular(e, m)
		cacheSet(clave, strconv.FormatFloat(prob, 'g', -1, 64), 10*time.Minute)
	}

	ms := float64(time.Since(inicio).Microseconds()) / 1000
	resp := RespuestaPrediccion{Probabilidad: prob, NivelRiesgo: nivel(prob), DesdeCache: desdeCache, TiempoMs: ms}
	registrarLatencia(ms)
	// guardamos en MongoDB en segundo plano (otra goroutine) para no demorar la respuesta
	go guardarPrediccion(Prediccion{Fecha: time.Now(), Entrada: e, Probabilidad: prob, Nivel: resp.NivelRiesgo,
		LatenciaMs: ms, DesdeCache: desdeCache})

	w.Header().Set("X-Tiempo-ms", strconv.FormatFloat(ms, 'f', 3, 64))
	responderJSON(w, 200, resp)
}

func entrenarHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		responderError(w, 405, "usa el método POST")
		return
	}
	pedido := struct {
		Iteraciones int     `json:"iteraciones"`
		LR          float64 `json:"lr"`
		Lambda      float64 `json:"lambda"`
	}{}
	if err := json.NewDecoder(r.Body).Decode(&pedido); err != nil && err != io.EOF {
		responderError(w, 400, "el cuerpo no es un JSON válido")
		return
	}
	if pedido.Iteraciones == 0 {
		pedido.Iteraciones = 100
	}
	if pedido.LR == 0 {
		pedido.LR = 0.5
	}
	if pedido.Lambda == 0 {
		pedido.Lambda = 0.00001
	}
	if pedido.Iteraciones < 1 || pedido.Iteraciones > 1000 {
		responderError(w, 400, "iteraciones debe estar entre 1 y 1000")
		return
	}
	id := time.Now().Format("20060102-150405")
	if !reservarEntrenamiento(id, pedido.Iteraciones) {
		responderError(w, 409, "ya hay un entrenamiento en curso")
		return
	}
	go entrenar(id, pedido.Iteraciones, pedido.LR, pedido.Lambda) // se entrena en segundo plano
	responderJSON(w, 202, map[string]string{"id": id, "mensaje": "entrenamiento iniciado; consulta /api/entrenar/estado"})
}

func estadoHandler(w http.ResponseWriter, r *http.Request) {
	responderJSON(w, 200, verEstado())
}

func entrenamientosHandler(w http.ResponseWriter, r *http.Request) {
	if colEntrenamientos == nil {
		responderError(w, 503, "MongoDB no está disponible")
		return
	}
	lista, err := listarEntrenamientos(10)
	if err != nil {
		responderError(w, 500, err.Error())
		return
	}
	responderJSON(w, 200, lista)
}

func nodosHandler(w http.ResponseWriter, r *http.Request) {
	responderJSON(w, 200, revisarNodos())
}

func prediccionesHandler(w http.ResponseWriter, r *http.Request) {
	if colPredicciones == nil {
		responderError(w, 503, "MongoDB no está disponible")
		return
	}
	lista, err := listarPredicciones(20)
	if err != nil {
		responderError(w, 500, err.Error())
		return
	}
	responderJSON(w, 200, lista)
}

func metricasHandler(w http.ResponseWriter, r *http.Request) {
	var ultimo interface{}
	if texto, ok := cacheGet("ultimo_entrenamiento"); ok {
		var e Entrenamiento
		if json.Unmarshal([]byte(texto), &e) == nil {
			e.Pesos = nil
			ultimo = e
		}
	}
	_, hay := modeloActual()
	responderJSON(w, 200, map[string]interface{}{
		"latencia_prediccion":    resumenLatencias(),
		"modelo_cargado":         hay,
		"ultimo_entrenamiento":   ultimo,
		"entrenamiento_en_curso": verEstado(),
	})
}

const paginaSwagger = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>API Órdenes de Compra</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url: "/openapi.yaml", dom_id: "#swagger-ui"});</script>
</body></html>`

// Permite que otras páginas (como el futuro frontend) llamen a la API
func cors(siguiente http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		siguiente.ServeHTTP(w, r)
	})
}

func main() {
	// Conectar con las bases de datos (se reintenta porque pueden demorar en arrancar)
	for intento := 1; intento <= 15; intento++ {
		err := conectarBD()
		if err == nil {
			fmt.Println("conectado a MongoDB y Redis")
			break
		}
		fmt.Printf("esperando a las bases de datos (%d/15): %v\n", intento, err)
		if intento == 15 {
			fmt.Println("AVISO: sin bases de datos; la API funcionará pero no guardará el historial")
			break
		}
		time.Sleep(2 * time.Second)
	}

	// Cargar el modelo que dejó ml_engine.go
	if m, err := cargarModelo(entorno("MODELO", "modelo.txt")); err != nil {
		fmt.Println("AVISO: no se cargó el modelo:", err)
	} else {
		establecerModelo(m)
		fmt.Println("modelo cargado correctamente")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", health)
	mux.HandleFunc("/api/predecir", predecir)
	mux.HandleFunc("/api/entrenar", entrenarHandler)
	mux.HandleFunc("/api/entrenar/estado", estadoHandler)
	mux.HandleFunc("/api/entrenamientos", entrenamientosHandler)
	mux.HandleFunc("/api/nodos", nodosHandler)
	mux.HandleFunc("/api/predicciones", prediccionesHandler)
	mux.HandleFunc("/api/metricas", metricasHandler)
	mux.HandleFunc("/openapi.yaml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml")
		w.Write(openapiYAML)
	})
	mux.HandleFunc("/swagger/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, paginaSwagger)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		http.Redirect(w, r, "/swagger/", http.StatusFound)
	})

	puerto := entorno("PUERTO", "8080")
	fmt.Println("API escuchando en el puerto", puerto, "- documentación en /swagger/")
	if err := http.ListenAndServe(":"+puerto, cors(mux)); err != nil {
		fmt.Println("error del servidor:", err)
		os.Exit(1)
	}
}
