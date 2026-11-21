package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"compras/cluster/ml"
)

// Una fila del dataset
type Fila struct {
	X [ml.NF]float32
	Y uint8
}

// Los datos de este nodo (se cargan una sola vez al arrancar)
var train, val, test []Fila

func entorno(clave, defecto string) string {
	if v := os.Getenv(clave); v != "" {
		return v
	}
	return defecto
}

// Lee el CSV y se queda solo con las filas que le tocan a este nodo
func cargarDatos(ruta string, id, total int) error {
	f, err := os.Open(ruta)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	sc.Scan() // encabezado
	linea := 0
	for sc.Scan() {
		linea++
		if (linea-1)%total != id { // esta fila le toca a otro nodo
			continue
		}
		campos := strings.Split(sc.Text(), ",")
		if len(campos) < 3+ml.NF || campos[2] == "1" { // campos[2] = masiva: no se usa
			continue
		}
		var fila Fila
		y, _ := strconv.Atoi(campos[1])
		fila.Y = uint8(y)
		for j := 0; j < ml.NF; j++ {
			v, _ := strconv.ParseFloat(campos[3+j], 32)
			fila.X[j] = float32(v)
		}
		switch campos[0] {
		case "0":
			train = append(train, fila)
		case "1":
			val = append(val, fila)
		default:
			test = append(test, fila)
		}
	}
	return sc.Err()
}

// Convierte una fila a float64 y estandariza (si ya se recibieron la media y la desviación)
func preparar(f *Fila, media, desv []float64) [ml.NF]float64 {
	var x [ml.NF]float64
	for j := 0; j < ml.NF; j++ {
		x[j] = float64(f.X[j])
	}
	if media != nil {
		ml.Estandarizar(&x, media, desv)
	}
	return x
}

// Suma y suma de cuadrados de las variables continuas (para que el coordinador calcule media y desviación)
func estadisticas() ml.Respuesta {
	k := len(ml.ColsContinuas)
	suma := make([]float64, k)
	sumaCuad := make([]float64, k)
	positivos := 0
	for i := range train {
		for c, j := range ml.ColsContinuas {
			v := float64(train[i].X[j])
			suma[c] += v
			sumaCuad[c] += v * v
		}
		if train[i].Y == 1 {
			positivos++
		}
	}
	return ml.Respuesta{N: len(train), Positivos: positivos, Suma: suma, SumaCuad: sumaCuad}
}

// Gradiente de un pedazo de filas
func gradienteParcial(datos []Fila, w, media, desv []float64, pesoPos float64) ml.Respuesta {
	grad := make([]float64, ml.NF+1)
	var sumaPesos, perdida float64
	for i := range datos {
		x := preparar(&datos[i], media, desv)
		p := ml.Sigmoide(ml.Puntaje(w, &x))
		y := float64(datos[i].Y)
		peso := 1.0
		if datos[i].Y == 1 {
			peso = pesoPos
		}
		dif := peso * (p - y)
		for j := 0; j < ml.NF; j++ {
			grad[j] += dif * x[j]
		}
		grad[ml.NF] += dif
		sumaPesos += peso
		if datos[i].Y == 1 {
			perdida += peso * -math.Log(math.Max(p, 1e-12))
		} else {
			perdida += -math.Log(math.Max(1-p, 1e-12))
		}
	}
	return ml.Respuesta{Grad: grad, Peso: sumaPesos, Perdida: perdida}
}

// Reparte las filas del nodo entre varias goroutines (una por núcleo) y junta los resultados
func gradiente(datos []Fila, w, media, desv []float64, pesoPos float64) ml.Respuesta {
	p := runtime.NumCPU()
	parciales := make([]ml.Respuesta, p) // cada goroutine escribe solo en su posición
	tam := (len(datos) + p - 1) / p
	var wg sync.WaitGroup
	for k := 0; k < p; k++ {
		ini, fin := k*tam, (k+1)*tam
		if ini > len(datos) {
			ini = len(datos)
		}
		if fin > len(datos) {
			fin = len(datos)
		}
		wg.Add(1)
		go func(k, ini, fin int) {
			defer wg.Done()
			parciales[k] = gradienteParcial(datos[ini:fin], w, media, desv, pesoPos)
		}(k, ini, fin)
	}
	wg.Wait()

	total := ml.Respuesta{Grad: make([]float64, ml.NF+1)}
	for _, r := range parciales {
		for j := range total.Grad {
			total.Grad[j] += r.Grad[j]
		}
		total.Peso += r.Peso
		total.Perdida += r.Perdida
	}
	return total
}

// Cuenta cuántas anuladas y no anuladas hay en cada casillero de puntaje (validación o test)
func evaluar(split string, w, media, desv []float64) ml.Respuesta {
	datos := val
	if split == "test" {
		datos = test
	}
	pos := make([]int, ml.Bins)
	neg := make([]int, ml.Bins)
	for i := range datos {
		x := preparar(&datos[i], media, desv)
		b := ml.Bin(ml.Puntaje(w, &x))
		if datos[i].Y == 1 {
			pos[b]++
		} else {
			neg[b]++
		}
	}
	return ml.Respuesta{N: len(datos), HistPos: pos, HistNeg: neg}
}

// Atiende a un coordinador conectado: lee mensajes y responde uno por uno
func atender(conn net.Conn) {
	defer conn.Close()
	dec := json.NewDecoder(conn)
	enc := json.NewEncoder(conn)
	var media, desv []float64 // se guardan para esta conexión cuando llega "normalizar"
	for {
		var m ml.Mensaje
		if err := dec.Decode(&m); err != nil {
			return // el coordinador cerró la conexión
		}
		var r ml.Respuesta
		switch m.Tipo {
		case "ping":
			r = ml.Respuesta{N: len(train)}
		case "stats":
			r = estadisticas()
		case "normalizar":
			media, desv = m.Media, m.Desv
		case "gradiente":
			r = gradiente(train, m.W, media, desv, m.PesoPos)
		case "evaluar":
			r = evaluar(m.Split, m.W, media, desv)
		default:
			r.Error = "mensaje desconocido: " + m.Tipo
		}
		if err := enc.Encode(&r); err != nil {
			return
		}
	}
}

func main() {
	puerto := entorno("PUERTO", "9001")
	id, _ := strconv.Atoi(entorno("NODO_ID", "0"))
	total, _ := strconv.Atoi(entorno("TOTAL_NODOS", "1"))
	ruta := entorno("DATASET", "dataset_limpio.csv")
	if total < 1 {
		total = 1
	}

	fmt.Printf("nodo %d de %d: cargando %s ...\n", id, total, ruta)
	if err := cargarDatos(ruta, id, total); err != nil {
		fmt.Println("error cargando los datos:", err)
		os.Exit(1)
	}
	fmt.Printf("nodo %d listo: train %d | validación %d | test %d | goroutines por cálculo: %d\n",
		id, len(train), len(val), len(test), runtime.NumCPU())

	ln, err := net.Listen("tcp", ":"+puerto)
	if err != nil {
		fmt.Println("no se pudo abrir el puerto", puerto, err)
		os.Exit(1)
	}
	fmt.Println("escuchando en el puerto", puerto)
	for {
		conn, err := ln.Accept()
		if err != nil {
			continue
		}
		go atender(conn) // cada coordinador conectado se atiende en su propia goroutine
	}
}
