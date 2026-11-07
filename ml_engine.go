// Algoritmo: regresión logística entrenada con descenso de gradiente.
//
// Cómo ejecutarlo (primero hay que correr el cargador para tener dataset_limpio.csv):
//   go run ml_engine.go -datos dataset_limpio.csv -nodos 4
//   go run ml_engine.go -datos dataset_limpio.csv -nodos 4 -verificar
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const NF = 23 // cantidad de variables; debe ser igual que en el cargador

// Un conjunto de datos (train, validación o test).
type Conjunto struct {
	X []float64
	Y []float64
	N int
}

// Lo que cada nodo le devuelve al coordinador
type Resultado struct {
	Grad    []float64 // gradiente de su pedazo
	Peso    float64   // suma de los pesos de las filas de su pedazo
	Perdida float64
}

// -----------------------
// Leer el dataset 
// -----------------------

func cargar(ruta string) (train, val, test *Conjunto) {
	archivo, err := os.Open(ruta)
	if err != nil {
		fmt.Println("no se pudo abrir", ruta, err)
		os.Exit(1)
	}
	defer archivo.Close()

	train, val, test = &Conjunto{}, &Conjunto{}, &Conjunto{}
	lector := bufio.NewScanner(archivo)
	lector.Buffer(make([]byte, 1024*1024), 1024*1024)
	lector.Scan() // saltamos el encabezado
	for lector.Scan() {
		campos := strings.Split(lector.Text(), ",")
		if len(campos) < 3+NF {
			continue
		}
		if campos[2] == "1" { // anulación masiva: no se usa
			continue
		}
		var c *Conjunto
		switch campos[0] {
		case "0":
			c = train
		case "1":
			c = val
		default:
			c = test
		}
		y, _ := strconv.ParseFloat(campos[1], 64)
		c.Y = append(c.Y, y)
		for j := 0; j < NF; j++ {
			v, _ := strconv.ParseFloat(campos[3+j], 64)
			c.X = append(c.X, v)
		}
	}
	train.N, val.N, test.N = len(train.Y), len(val.Y), len(test.Y)
	return
}

// Estandariza solo las variables continuas 
// usando la media y la desviación del conjunto de entrenamiento.
func estandarizar(train, val, test *Conjunto) (media, desv [2]float64) {
	for j := 0; j < 2; j++ {
		suma, sumaCuad := 0.0, 0.0
		for i := 0; i < train.N; i++ {
			v := train.X[i*NF+j]
			suma += v
			sumaCuad += v * v
		}
		media[j] = suma / float64(train.N)
		desv[j] = math.Sqrt(sumaCuad/float64(train.N) - media[j]*media[j])
		if desv[j] == 0 {
			desv[j] = 1
		}
		for _, c := range []*Conjunto{train, val, test} {
			for i := 0; i < c.N; i++ {
				c.X[i*NF+j] = (c.X[i*NF+j] - media[j]) / desv[j]
			}
		}
	}
	return
}

// ----------------------------------------------
// Gradiente de un pedazo de datos
// ----------------------------------------------

func gradienteParcial(c *Conjunto, ini, fin int, w []float64, pesoPos float64) Resultado {
	grad := make([]float64, NF+1) // los NF pesos y el sesgo (al final)
	var sumaPesos, perdida float64
	for i := ini; i < fin; i++ {
		x := c.X[i*NF : (i+1)*NF]
		z := w[NF] // sesgo
		for j := 0; j < NF; j++ {
			z += w[j] * x[j]
		}
		p := 1 / (1 + math.Exp(-z))

		peso := 1.0
		if c.Y[i] == 1 {
			peso = pesoPos
		}
		dif := peso * (p - c.Y[i])
		for j := 0; j < NF; j++ {
			grad[j] += dif * x[j]
		}
		grad[NF] += dif
		sumaPesos += peso

		if c.Y[i] == 1 {
			perdida += peso * -math.Log(math.Max(p, 1e-12))
		} else {
			perdida += -math.Log(math.Max(1-p, 1e-12))
		}
	}
	return Resultado{Grad: grad, Peso: sumaPesos, Perdida: perdida}
}

// ---------------------------------------------------------------------------
// Un NODO: espera los pesos por un channel, calcula el gradiente de su pedazo
// y devuelve el resultado por otro channel.
// ---------------------------------------------------------------------------

func nodo(c *Conjunto, ini, fin int, pesoPos float64, entrada <-chan []float64, salida chan<- Resultado) {
	for w := range entrada { // se queda esperando pesos hasta que cierren el channel
		salida <- gradienteParcial(c, ini, fin, w, pesoPos)
	}
}

// ---------------------------------------------------------------------------
// El COORDINADOR: reparte el trabajo, junta los gradientes y actualiza los pesos
// ---------------------------------------------------------------------------

func entrenar(train *Conjunto, numNodos, iteraciones int, lr, lambda, pesoPos float64) []float64 {
	// 1) cada nodo se queda con un pedazo de las filas
	entradas := make([]chan []float64, numNodos)
	salida := make(chan Resultado, numNodos)
	tam := (train.N + numNodos - 1) / numNodos
	for k := 0; k < numNodos; k++ {
		ini := k * tam
		fin := ini + tam
		if fin > train.N {
			fin = train.N
		}
		if ini > fin {
			ini = fin
		}
		entradas[k] = make(chan []float64)
		go nodo(train, ini, fin, pesoPos, entradas[k], salida)
	}

	// 2) bucle de entrenamiento
	w := make([]float64, NF+1) // empezamos con todos los pesos en 0
	for it := 1; it <= iteraciones; it++ {
		// a) mandar los pesos actuales a todos los nodos
		for k := 0; k < numNodos; k++ {
			entradas[k] <- w
		}
		// b) esperar el resultado de TODOS los nodos y sumarlos
		grad := make([]float64, NF+1)
		var pesoTotal, perdida float64
		for k := 0; k < numNodos; k++ {
			r := <-salida
			for j := range grad {
				grad[j] += r.Grad[j]
			}
			pesoTotal += r.Peso
			perdida += r.Perdida
		}
		// c) actualizar los pesos (solo el coordinador los modifica)
		for j := range w {
			g := grad[j] / pesoTotal
			if j < NF { // regularización L2 (no se aplica al sesgo)
				g += lambda * w[j]
			}
			w[j] -= lr * g
		}
		if it == 1 || it%10 == 0 {
			fmt.Printf("iteración %3d | pérdida %.5f\n", it, perdida/pesoTotal)
		}
	}

	// 3) cerrar los channels para que las goroutines de los nodos terminen
	for k := 0; k < numNodos; k++ {
		close(entradas[k])
	}
	return w
}

// --------------------------------------------------------------
// Evaluación: ordenamos las órdenes de mayor a menor probabilidad
// -------------------------------------------------------------

func evaluar(c *Conjunto, w []float64, nombre string) {
	n := c.N
	puntaje := make([]float64, n)
	positivos := 0.0
	for i := 0; i < n; i++ {
		x := c.X[i*NF : (i+1)*NF]
		z := w[NF]
		for j := 0; j < NF; j++ {
			z += w[j] * x[j]
		}
		puntaje[i] = z 
		positivos += c.Y[i]
	}
	orden := make([]int, n)
	for i := range orden {
		orden[i] = i
	}
	sort.Slice(orden, func(a, b int) bool { return puntaje[orden[a]] > puntaje[orden[b]] })

	corte := n / 20 // el 5 % de órdenes con mayor riesgo
	if corte < 1 {
		corte = 1
	}
	var aciertos, ap, recall5, prec5 float64
	for pos, i := range orden {
		if c.Y[i] == 1 {
			aciertos++
			ap += aciertos / float64(pos+1) // precisión cada vez que encontramos una anulada
		}
		if pos+1 == corte {
			recall5 = aciertos / positivos
			prec5 = aciertos / float64(corte)
		}
	}
	ap /= positivos
	base := positivos / float64(n)

	fmt.Println("----", nombre, "----")
	fmt.Printf("órdenes: %d | anuladas: %.0f (%.2f %%)\n", n, positivos, base*100)
	fmt.Printf("AUC-PR: %.4f (un modelo sin información daría %.4f)\n", ap, base)
	fmt.Printf("revisando solo el 5 %% de mayor riesgo: se encuentra el %.1f %% de las anuladas (precisión %.2f %%, %.1f veces mejor que al azar)\n",
		recall5*100, prec5*100, prec5/base)
}

// ---------------------------------------------------------------------------
// Programa principal
// ---------------------------------------------------------------------------

func main() {
	datos := flag.String("datos", "dataset_limpio.csv", "archivo generado por el cargador")
	numNodos := flag.Int("nodos", 4, "cantidad de nodos (goroutines) que reparten el cálculo")
	iteraciones := flag.Int("iteraciones", 200, "iteraciones del descenso de gradiente")
	lr := flag.Float64("lr", 0.5, "tasa de aprendizaje")
	lambda := flag.Float64("lambda", 0.00001, "regularización L2")
	verificar := flag.Bool("verificar", false, "comprobar que 1 nodo y K nodos dan el mismo resultado")
	flag.Parse()

	fmt.Println("núcleos del procesador:", runtime.NumCPU())
	inicio := time.Now()
	train, val, test := cargar(*datos)
	fmt.Printf("datos cargados en %.1f s | train %d | validación %d | test %d\n",
		time.Since(inicio).Seconds(), train.N, val.N, test.N)
	estandarizar(train, val, test)

	// Peso de la clase positiva = cantidad de no anuladas / cantidad de anuladas
	positivos := 0.0
	for _, y := range train.Y {
		positivos += y
	}
	pesoPos := (float64(train.N) - positivos) / positivos
	fmt.Printf("anuladas en train: %.0f (%.2f %%) | peso de las anuladas: %.1f\n",
		positivos, positivos/float64(train.N)*100, pesoPos)

	// Modo verificación: con pocas iteraciones, 1 nodo y K nodos deben dar los mismos pesos
	if *verificar {
		w1 := entrenar(train, 1, 5, *lr, *lambda, pesoPos)
		wk := entrenar(train, *numNodos, 5, *lr, *lambda, pesoPos)
		maxDif := 0.0
		for j := range w1 {
			maxDif = math.Max(maxDif, math.Abs(w1[j]-wk[j]))
		}
		fmt.Printf("diferencia máxima entre 1 nodo y %d nodos: %.3e\n", *numNodos, maxDif)
		if maxDif < 1e-9 {
			fmt.Println("RESULTADO: OK, el cálculo repartido da lo mismo que el secuencial")
		} else {
			fmt.Println("RESULTADO: hay diferencia, revisar")
		}
		return
	}

	// Entrenamiento normal
	fmt.Println("entrenando con", *numNodos, "nodos...")
	inicioEntrenamiento := time.Now()
	w := entrenar(train, *numNodos, *iteraciones, *lr, *lambda, pesoPos)
	fmt.Printf("tiempo de entrenamiento: %.1f segundos con %d nodos\n", time.Since(inicioEntrenamiento).Seconds(), *numNodos)

	evaluar(val, w, "VALIDACIÓN")
	evaluar(test, w, "TEST")

	// Guardar los pesos del modelo
	f, err := os.Create("modelo.txt")
	if err == nil {
		for j, v := range w {
			fmt.Fprintf(f, "w%d=%g\n", j, v)
		}
		f.Close()
		fmt.Println("pesos guardados en modelo.txt")
	}
}
