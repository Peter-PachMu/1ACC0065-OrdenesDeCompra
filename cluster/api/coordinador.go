package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net"
	"strings"
	"sync"
	"time"

	"compras/cluster/ml"
)

// ---------------------------------------------------------------------------
// Conexión con los nodos
// ---------------------------------------------------------------------------

type Nodo struct {
	Direccion string
	conn      net.Conn
	enc       *json.Encoder
	dec       *json.Decoder
}

func conectarNodo(direccion string) (*Nodo, error) {
	conn, err := net.DialTimeout("tcp", direccion, 3*time.Second)
	if err != nil {
		return nil, err
	}
	return &Nodo{Direccion: direccion, conn: conn, enc: json.NewEncoder(conn), dec: json.NewDecoder(conn)}, nil
}

// Envía un mensaje al nodo y espera su respuesta
func (n *Nodo) pedir(m ml.Mensaje) (ml.Respuesta, error) {
	n.conn.SetDeadline(time.Now().Add(2 * time.Minute)) // si el nodo no responde, no esperamos para siempre
	if err := n.enc.Encode(&m); err != nil {
		return ml.Respuesta{}, fmt.Errorf("%s: %v", n.Direccion, err)
	}
	var r ml.Respuesta
	if err := n.dec.Decode(&r); err != nil {
		return r, fmt.Errorf("%s: %v", n.Direccion, err)
	}
	if r.Error != "" {
		return r, fmt.Errorf("%s: %s", n.Direccion, r.Error)
	}
	return r, nil
}

func (n *Nodo) cerrar() { n.conn.Close() }

func direccionesNodos() []string {
	direcciones := []string{}
	for _, d := range strings.Split(entorno("NODOS", "localhost:9001"), ",") {
		if d = strings.TrimSpace(d); d != "" {
			direcciones = append(direcciones, d)
		}
	}
	return direcciones
}

func conectarTodos() ([]*Nodo, error) {
	nodos := []*Nodo{}
	for _, d := range direccionesNodos() {
		n, err := conectarNodo(d)
		if err != nil {
			for _, x := range nodos {
				x.cerrar()
			}
			return nil, fmt.Errorf("no se pudo conectar con el nodo %s: %v", d, err)
		}
		nodos = append(nodos, n)
	}
	return nodos, nil
}

// Envía el mismo mensaje a todos los nodos AL MISMO TIEMPO y espera todas las respuestas
func pedirATodos(nodos []*Nodo, m ml.Mensaje) ([]ml.Respuesta, error) {
	respuestas := make([]ml.Respuesta, len(nodos))
	errores := make([]error, len(nodos))
	var wg sync.WaitGroup
	for i, n := range nodos {
		wg.Add(1)
		go func(i int, n *Nodo) {
			defer wg.Done()
			respuestas[i], errores[i] = n.pedir(m) // cada goroutine escribe solo en su posición
		}(i, n)
	}
	wg.Wait()
	for _, e := range errores {
		if e != nil {
			return nil, e
		}
	}
	return respuestas, nil
}

// Revisa qué nodos están activos y cuánto demoran en responder
func revisarNodos() []EstadoNodo {
	direcciones := direccionesNodos()
	estados := make([]EstadoNodo, len(direcciones))
	var wg sync.WaitGroup
	for i, d := range direcciones {
		wg.Add(1)
		go func(i int, d string) {
			defer wg.Done()
			e := EstadoNodo{Direccion: d, Revisado: time.Now()}
			inicio := time.Now()
			if n, err := conectarNodo(d); err == nil {
				if r, err := n.pedir(ml.Mensaje{Tipo: "ping"}); err == nil {
					e.Activo = true
					e.MuestrasTrain = r.N
					e.LatenciaMs = float64(time.Since(inicio).Microseconds()) / 1000
				}
				n.cerrar()
			}
			estados[i] = e
		}(i, d)
	}
	wg.Wait()
	for _, e := range estados {
		guardarNodo(e)
	}
	return estados
}

// ---------------------------------------------------------------------------
// Progreso del entrenamiento (se consulta desde la API mientras corre)
// ---------------------------------------------------------------------------

type EstadoEntrenamiento struct {
	Corriendo bool    `json:"corriendo"`
	ID        string  `json:"id"`
	Iteracion int     `json:"iteracion"`
	Total     int     `json:"total_iteraciones"`
	Perdida   float64 `json:"perdida"`
	Mensaje   string  `json:"mensaje"`
}

var (
	estado        EstadoEntrenamiento
	candadoEstado sync.Mutex // protege "estado", porque lo usan varias goroutines
)

func verEstado() EstadoEntrenamiento {
	candadoEstado.Lock()
	defer candadoEstado.Unlock()
	return estado
}

func cambiarEstado(f func(e *EstadoEntrenamiento)) {
	candadoEstado.Lock()
	defer candadoEstado.Unlock()
	f(&estado)
}

// Intenta marcar que empieza un entrenamiento. Devuelve false si ya hay uno corriendo.
func reservarEntrenamiento(id string, iteraciones int) bool {
	ok := false
	cambiarEstado(func(e *EstadoEntrenamiento) {
		if !e.Corriendo {
			*e = EstadoEntrenamiento{Corriendo: true, ID: id, Total: iteraciones, Mensaje: "iniciando"}
			ok = true
		}
	})
	return ok
}

// ---------------------------------------------------------------------------
// Entrenamiento distribuido
// ---------------------------------------------------------------------------

func entrenar(id string, iteraciones int, lr, lambda float64) {
	ent := Entrenamiento{ID: id, Inicio: time.Now(), Estado: "en_curso", Iteraciones: iteraciones, LR: lr, Lambda: lambda}

	fallar := func(err error) {
		ent.Estado = "error: " + err.Error()
		ent.Fin = time.Now()
		ent.Segundos = ent.Fin.Sub(ent.Inicio).Seconds()
		cambiarEstado(func(e *EstadoEntrenamiento) { e.Corriendo = false; e.Mensaje = ent.Estado })
		guardarEntrenamiento(ent)
	}

	nodos, err := conectarTodos()
	if err != nil {
		fallar(err)
		return
	}
	defer func() {
		for _, n := range nodos {
			n.cerrar()
		}
	}()
	ent.Nodos = len(nodos)

	// --- Fase 0: estadísticas globales para estandarizar las variables continuas ---
	respuestas, err := pedirATodos(nodos, ml.Mensaje{Tipo: "stats"})
	if err != nil {
		fallar(err)
		return
	}
	k := len(ml.ColsContinuas)
	suma := make([]float64, k)
	sumaCuad := make([]float64, k)
	total, positivos := 0, 0
	for _, r := range respuestas {
		total += r.N
		positivos += r.Positivos
		for i := 0; i < k; i++ {
			suma[i] += r.Suma[i]
			sumaCuad[i] += r.SumaCuad[i]
		}
	}
	if positivos == 0 {
		fallar(fmt.Errorf("no hay órdenes anuladas en los datos de entrenamiento"))
		return
	}
	media := make([]float64, k)
	desv := make([]float64, k)
	for i := 0; i < k; i++ {
		media[i] = suma[i] / float64(total)
		desv[i] = math.Sqrt(sumaCuad[i]/float64(total) - media[i]*media[i])
		if desv[i] == 0 {
			desv[i] = 1
		}
	}
	if _, err := pedirATodos(nodos, ml.Mensaje{Tipo: "normalizar", Media: media, Desv: desv}); err != nil {
		fallar(err)
		return
	}
	pesoPos := float64(total-positivos) / float64(positivos)
	ent.PesoPos = pesoPos

	// --- Bucle de entrenamiento ---
	w := make([]float64, ml.NF+1) // empezamos con todos los pesos en 0
	var pesoTotal, perdida float64
	for it := 1; it <= iteraciones; it++ {
		resp, err := pedirATodos(nodos, ml.Mensaje{Tipo: "gradiente", W: w, PesoPos: pesoPos})
		if err != nil {
			fallar(err)
			return
		}
		grad := make([]float64, ml.NF+1)
		pesoTotal, perdida = 0, 0
		for _, r := range resp {
			for j := range grad {
				grad[j] += r.Grad[j]
			}
			pesoTotal += r.Peso
			perdida += r.Perdida
		}
		for j := range w {
			g := grad[j] / pesoTotal
			if j < ml.NF { // regularización L2 (no se aplica al sesgo)
				g += lambda * w[j]
			}
			w[j] -= lr * g
		}
		p := perdida / pesoTotal
		cambiarEstado(func(e *EstadoEntrenamiento) { e.Iteracion = it; e.Perdida = p })
	}

	// --- Métricas de validación y test (cada nodo manda su histograma y aquí se suman) ---
	evaluarSplit := func(split string) (ml.Metricas, error) {
		resp, err := pedirATodos(nodos, ml.Mensaje{Tipo: "evaluar", Split: split, W: w})
		if err != nil {
			return ml.Metricas{}, err
		}
		pos := make([]int, ml.Bins)
		neg := make([]int, ml.Bins)
		for _, r := range resp {
			for b := 0; b < ml.Bins; b++ {
				pos[b] += r.HistPos[b]
				neg[b] += r.HistNeg[b]
			}
		}
		return ml.MetricasDeHistograma(pos, neg), nil
	}
	mv, err := evaluarSplit("val")
	if err != nil {
		fallar(err)
		return
	}
	mt, err := evaluarSplit("test")
	if err != nil {
		fallar(err)
		return
	}

	// --- Guardar resultados ---
	ent.Estado = "terminado"
	ent.Fin = time.Now()
	ent.Segundos = ent.Fin.Sub(ent.Inicio).Seconds()
	ent.PerdidaFinal = perdida / pesoTotal
	ent.AUCPRVal = mv.AUCPR
	ent.AUCPRTest = mt.AUCPR
	ent.Recall5Test = mt.Recall5
	ent.Precision5Test = mt.Precision5
	ent.Lift5Test = mt.Lift5
	ent.Pesos = w

	establecerModelo(Modelo{W: w, Media: media, Desv: desv, PesoPos: pesoPos}) // la API usa el modelo nuevo
	guardarEntrenamiento(ent)
	if texto, err := json.Marshal(ent); err == nil {
		cacheSet("ultimo_entrenamiento", string(texto), 0)
	}
	cambiarEstado(func(e *EstadoEntrenamiento) { e.Corriendo = false; e.Mensaje = "terminado" })
	fmt.Printf("entrenamiento %s terminado en %.1f s | AUC-PR test %.4f | lift@5%% %.2f\n", id, ent.Segundos, mt.AUCPR, mt.Lift5)
}
