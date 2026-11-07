// Idea:
//   archivos .xlsx -> LECTORES -> filasCh -> WORKERS -> resultCh -> COLECTOR (main)
//                     (leen)               (validan y     (junta, quita
//                                           limpian)       duplicados, etc.)
//
// Salida: un solo archivo, dataset_limpio.csv, y un resumen en pantalla.
//
// Cómo ejecutarlo:
//   go mod init compras
//   go get github.com/xuri/excelize/v2
//   go run data_loader.go -in ./datos -out dataset_limpio.csv
package main

import (
	"bufio"
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/xuri/excelize/v2"
)

// Cantidad de variables (features) que se generan para cada orden:
//   x0 = log del monto            
//   x1 = largo de la descripción
//   x2 = es fin de mes            
//   x3 = contratista es persona jurídica (RUC empieza con 20)
//   x4..x6   = objeto contractual (bienes, servicios, consultoría/obras)
//   x7..x15  = tipo de contratación (9 tipos)
//   x16..x22 = sector (salud, educación, municipal, gob. regional, seguridad, saneamiento, otro)
const NF = 23

const mesesTest = 2 // los últimos 2 meses son para test
const mesesVal = 2  // los 2 meses anteriores son para validación


type Fila struct {
	Col   map[string]int 
	Datos []string
}


type Orden struct {
	RUC     int64 // RUC de la entidad
	Periodo int   // AAAAMM
	Anulada int   // 1 si el estado es ANULADA
	Masiva  int   // 1 si pertenece a una anulación masiva (se excluye luego del entrenamiento)
	Split   int   // 0 train, 1 validación, 2 test
	X       [NF]float32
}

// Lo que devuelve cada worker por cada fila
type Resultado struct {
	Motivo string // vacío si la fila es válida; si no, el motivo del descarte
	Llave  uint64 // sirve para detectar duplicados
	Orden  Orden
}

var quitarTildes = strings.NewReplacer("Á", "A", "É", "E", "Í", "I", "Ó", "O", "Ú", "U", "Ñ", "N")

func limpiarTexto(s string) string {
	return quitarTildes.Replace(strings.ToUpper(strings.TrimSpace(s)))
}

func contiene(s string, palabras ...string) bool {
	for _, p := range palabras {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

func hash(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// ---------------------------------------------------------------------------
// Funciones para convertir texto en códigos
// ---------------------------------------------------------------------------

func codObjeto(texto string) int {
	s := limpiarTexto(texto)
	if contiene(s, "CONSULTOR", "OBRA") {
		return 2
	}
	if contiene(s, "BIENES") {
		return 0
	}
	if contiene(s, "SERVICIO") {
		return 1
	}
	return -1
}

func codTipoContratacion(texto string) int {
	s := limpiarTexto(texto)
	if contiene(s, "HASTA 8 UIT") {
		return 0
	}
	if contiene(s, "SIN PROCESO") {
		return 1
	}
	if contiene(s, "SELECCION") {
		return 2
	}
	if contiene(s, "EXONERACION", "CONTRATACION DIRECTA") {
		return 3
	}
	if contiene(s, "CATALOGO") {
		return 4
	}
	if contiene(s, "REGIMENES ESPECIALES") {
		return 5
	}
	if contiene(s, "CONVENIO") {
		return 6
	}
	if contiene(s, "SERVICIO PUBLICO") {
		return 7
	}
	if contiene(s, "DECRETO") {
		return 8
	}
	return -1
}

func codSector(entidad string) int {
	s := limpiarTexto(entidad)
	if contiene(s, "SALUD", "HOSPITAL") {
		return 0
	}
	if contiene(s, "UGEL", "EDUCACI", "UNIVERSIDAD") {
		return 1
	}
	if contiene(s, "MUNICIPALIDAD") {
		return 2
	}
	if contiene(s, "GOBIERNO REGIONAL") {
		return 3
	}
	if contiene(s, "POLIC", "INTERIOR", "PODER JUDICIAL", "MINISTERIO PUBLICO", "EJERCITO", "MARINA") {
		return 4
	}
	if contiene(s, "SANEAMIENTO", "SEDAPAL") {
		return 5
	}
	return 6
}

func leerFecha(texto string) (time.Time, bool) {
	if len(texto) >= 10 {
		texto = texto[:10]
	}
	for _, formato := range []string{"2006-01-02", "02/01/2006"} {
		t, err := time.Parse(formato, texto)
		if err == nil && t.Year() >= 2020 && t.Year() <= 2030 {
			return t, true
		}
	}
	return time.Time{}, false
}

// ---------------------------------------------------------------------------
// ETAPA 1: lectura. Cada lector abre un archivo y manda sus filas al canal.
// ---------------------------------------------------------------------------

var columnasNecesarias = []string{"ENTIDAD", "RUC_ENTIDAD", "ORDEN", "DESCRIPCION_ORDEN", "OBJETOCONTRACTUAL",
	"ESTADOCONTRATACION", "TIPODECONTRATACION", "MONTO_TOTAL_ORDEN_ORIGINAL", "RUC_CONTRATISTA", "FECHA_DE_EMISION"}

func leerArchivo(ruta string, filasCh chan<- Fila) {
	f, err := excelize.OpenFile(ruta)
	if err != nil {
		fmt.Println("no se pudo abrir", ruta, err)
		return
	}
	defer f.Close()

	hojas := f.GetSheetList()
	if len(hojas) == 0 {
		fmt.Println("el archivo no tiene hojas:", ruta)
		return
	}
	filas, err := f.GetRows(hojas[0])
	if err != nil || len(filas) < 2 {
		fmt.Println("archivo vacío o con error:", ruta)
		return
	}

	// La primera fila es el encabezado: guardamos la posición de cada columna
	col := map[string]int{}
	for i, nombre := range filas[0] {
		col[strings.ToUpper(strings.TrimSpace(nombre))] = i
	}
	for _, c := range columnasNecesarias {
		if _, ok := col[c]; !ok {
			fmt.Println("al archivo", filepath.Base(ruta), "le falta la columna", c)
			return
		}
	}

	for _, fila := range filas[1:] {
		filasCh <- Fila{Col: col, Datos: fila}
	}
	fmt.Println("leído:", filepath.Base(ruta), "-", len(filas)-1, "filas")
}

// ----------------------------------------------------
// ETAPA 2: validación y limpieza de una fila 
// -----------------------------------------------
func procesar(f Fila) Resultado {
	dato := func(nombre string) string {
		i := f.Col[nombre]
		if i >= len(f.Datos) {
			return ""
		}
		return strings.TrimSpace(f.Datos[i])
	}

	// 1) el monto debe ser un número mayor que 0
	montoTxt := dato("MONTO_TOTAL_ORDEN_ORIGINAL")
	monto, err := strconv.ParseFloat(strings.ReplaceAll(montoTxt, ",", ""), 64)
	if err != nil || monto <= 0 {
		return Resultado{Motivo: "monto invalido"}
	}

	// 2) el estado debe ser uno de los 4 conocidos
	estado := limpiarTexto(dato("ESTADOCONTRATACION"))
	if estado != "ANULADA" && estado != "COMPROMETIDA" && estado != "DEVENGADA" && estado != "EMITIDA" {
		return Resultado{Motivo: "estado invalido"}
	}

	// 3) la fecha de emisión debe ser válida
	fechaTxt := dato("FECHA_DE_EMISION")
	fecha, ok := leerFecha(fechaTxt)
	if !ok {
		return Resultado{Motivo: "fecha invalida"}
	}

	rucEntidad := dato("RUC_ENTIDAD")
	rucContratista := dato("RUC_CONTRATISTA")
	ruc, _ := strconv.ParseInt(rucEntidad, 10, 64)

	// Variables para el modelo
	var x [NF]float32
	x[0] = float32(math.Log1p(monto))
	x[1] = float32(len(dato("DESCRIPCION_ORDEN")))
	if fecha.AddDate(0, 0, 3).Month() != fecha.Month() { // estamos en los últimos 3 días del mes
		x[2] = 1
	}
	if strings.HasPrefix(rucContratista, "20") {
		x[3] = 1
	}
	if c := codObjeto(dato("OBJETOCONTRACTUAL")); c >= 0 {
		x[4+c] = 1
	}
	if c := codTipoContratacion(dato("TIPODECONTRATACION")); c >= 0 {
		x[7+c] = 1
	}
	x[16+codSector(dato("ENTIDAD"))] = 1

	o := Orden{RUC: ruc, Periodo: fecha.Year()*100 + int(fecha.Month()), X: x}
	if estado == "ANULADA" {
		o.Anulada = 1
	}

	// Llave para detectar duplicados
	llave := hash(rucEntidad + "|" + dato("ORDEN") + "|" + rucContratista + "|" + fechaTxt + "|" + montoTxt + "|" + estado)
	return Resultado{Llave: llave, Orden: o}
}

// 
var reAnio = regexp.MustCompile("20[0-9][0-9]")

func periodoDelArchivo(nombre string) int {
	meses := map[string]int{"ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6, "JULIO": 7,
		"AGOSTO": 8, "SETIEMBRE": 9, "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12}
	n := strings.ToUpper(nombre)
	anio := reAnio.FindString(n)
	if anio == "" {
		return 0
	}
	a, _ := strconv.Atoi(anio)
	for nombreMes, numero := range meses {
		if strings.Contains(n, nombreMes) {
			return a*100 + numero
		}
	}
	return 0
}

// ------------------------
// Programa principal
// ------------------------

func main() {
	carpeta := flag.String("in", "./datos", "carpeta con los archivos mensuales (.xlsx)")
	salida := flag.String("out", "dataset_limpio.csv", "archivo de salida")
	numWorkers := flag.Int("workers", runtime.NumCPU(), "goroutines que limpian filas (por defecto, los núcleos del procesador)")
	numLectores := flag.Int("lectores", 2, "archivos que se leen al mismo tiempo")
	reservar := flag.String("reservar", "202609,202610", "periodos AAAAMM que NO se usan (pruebas en vivo)")
	flag.Parse()
	inicio := time.Now()

	// --- Buscar los archivos saltando los periodos reservados ---
	reservados := map[int]bool{}
	for _, p := range strings.Split(*reservar, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err == nil {
			reservados[n] = true
		}
	}
	archivos := []string{}
	todos, _ := filepath.Glob(filepath.Join(*carpeta, "ORDENESCOMPRA*.xlsx"))
	for _, ruta := range todos {
		if reservados[periodoDelArchivo(filepath.Base(ruta))] {
			fmt.Println("reservado (no se lee):", filepath.Base(ruta))
			continue
		}
		archivos = append(archivos, ruta)
	}
	if len(archivos) == 0 {
		fmt.Println("no se encontraron archivos ORDENESCOMPRA*.xlsx en", *carpeta)
		return
	}
	fmt.Println("archivos a leer:", len(archivos), "| workers:", *numWorkers, "| lectores:", *numLectores)

	// --- Pipeline concurrente ---
	filasCh := make(chan Fila, 1000)
	resultCh := make(chan Resultado, 1000)
	rutasCh := make(chan string)

	// Lectores: varias goroutines que toman rutas del canal y leen los archivos
	var wgLectores sync.WaitGroup
	for i := 0; i < *numLectores; i++ {
		wgLectores.Add(1)
		go func() {
			defer wgLectores.Done()
			for ruta := range rutasCh {
				leerArchivo(ruta, filasCh)
			}
		}()
	}
	go func() { // reparte las rutas entre los lectores
		for _, r := range archivos {
			rutasCh <- r
		}
		close(rutasCh)
	}()
	go func() { // cuando todos los lectores terminan, se cierra el canal de filas
		wgLectores.Wait()
		close(filasCh)
	}()

	// Workers: varias goroutines que toman filas, las limpian y mandan el resultado
	var wgWorkers sync.WaitGroup
	for i := 0; i < *numWorkers; i++ {
		wgWorkers.Add(1)
		go func() {
			defer wgWorkers.Done()
			for fila := range filasCh {
				resultCh <- procesar(fila)
			}
		}()
	}
	go func() { // cuando todos los workers terminan, se cierra el canal de resultados
		wgWorkers.Wait()
		close(resultCh)
	}()

	// Colector: junta los resultados y quita duplicados.
	// Como solo el main toca el mapa "vistos", no hace falta ningún mutex.
	vistos := map[uint64]bool{}
	ordenes := []Orden{}
	descartes := map[string]int{}
	total := 0
	for r := range resultCh {
		total++
		if r.Motivo != "" {
			descartes[r.Motivo]++
			continue
		}
		if reservados[r.Orden.Periodo] {
			descartes["periodo reservado"]++
			continue
		}
		if vistos[r.Llave] {
			descartes["duplicada"]++
			continue
		}
		vistos[r.Llave] = true
		ordenes = append(ordenes, r.Orden)
	}

	// --- Marcar anulaciones masivas ---
	// Si una entidad, en un mes, tiene 30 o más órdenes y el 90 % o más están anuladas,
	// consideramos que es una anulación masiva (administrativa) y no la usamos para entrenar.
	type clave struct {
		ruc     int64
		periodo int
	}
	totalPorEM := map[clave]int{}
	anuladasPorEM := map[clave]int{}
	for _, o := range ordenes {
		k := clave{o.RUC, o.Periodo}
		totalPorEM[k]++
		if o.Anulada == 1 {
			anuladasPorEM[k]++
		}
	}
	for i := range ordenes {
		k := clave{ordenes[i].RUC, ordenes[i].Periodo}
		if totalPorEM[k] >= 30 && float64(anuladasPorEM[k])/float64(totalPorEM[k]) >= 0.9 {
			ordenes[i].Masiva = 1
		}
	}

	// --- Partición por fecha: train / validación / test ---
	vistoP := map[int]bool{}
	periodos := []int{}
	for _, o := range ordenes {
		if !vistoP[o.Periodo] {
			vistoP[o.Periodo] = true
			periodos = append(periodos, o.Periodo)
		}
	}
	sort.Ints(periodos)
	if len(periodos) <= mesesTest+mesesVal {
		fmt.Println("se necesitan más de", mesesTest+mesesVal, "meses de datos")
		return
	}
	splitDe := map[int]int{}
	for i, p := range periodos {
		if i >= len(periodos)-mesesTest {
			splitDe[p] = 2
		} else if i >= len(periodos)-mesesTest-mesesVal {
			splitDe[p] = 1
		} else {
			splitDe[p] = 0
		}
	}
	for i := range ordenes {
		ordenes[i].Split = splitDe[ordenes[i].Periodo]
	}

	// --- Guardar el dataset limpio ---
	f, err := os.Create(*salida)
	if err != nil {
		fmt.Println("no se pudo crear", *salida, err)
		return
	}
	w := bufio.NewWriter(f)
	fmt.Fprint(w, "split,y,masiva")
	for j := 0; j < NF; j++ {
		fmt.Fprintf(w, ",x%d", j)
	}
	fmt.Fprintln(w)
	for _, o := range ordenes {
		fmt.Fprintf(w, "%d,%d,%d", o.Split, o.Anulada, o.Masiva)
		for j := 0; j < NF; j++ {
			fmt.Fprintf(w, ",%g", o.X[j])
		}
		fmt.Fprintln(w)
	}
	w.Flush()
	f.Close()

	// --- Resumen en pantalla---
	anuladas, masivas := 0, 0
	var cuenta [3]int
	var cuentaAnuladas [3]int
	for _, o := range ordenes {
		anuladas += o.Anulada
		if o.Masiva == 1 {
			masivas += o.Anulada
			continue
		}
		cuenta[o.Split]++
		cuentaAnuladas[o.Split] += o.Anulada
	}
	fmt.Println()
	fmt.Println("========== RESUMEN ==========")
	fmt.Println("filas leídas:            ", total)
	for motivo, n := range descartes {
		fmt.Println("  descartadas por", motivo+":", n)
	}
	fmt.Println("filas limpias:           ", len(ordenes))
	fmt.Println("órdenes anuladas:        ", anuladas)
	fmt.Println("  de ellas, masivas:     ", masivas)
	fmt.Println("train      (sin masivas):", cuenta[0], "órdenes,", cuentaAnuladas[0], "anuladas")
	fmt.Println("validación (sin masivas):", cuenta[1], "órdenes,", cuentaAnuladas[1], "anuladas")
	fmt.Println("test       (sin masivas):", cuenta[2], "órdenes,", cuentaAnuladas[2], "anuladas")
	fmt.Printf("tiempo total: %.1f segundos\n", time.Since(inicio).Seconds())
	fmt.Println("archivo generado:", *salida)
}
