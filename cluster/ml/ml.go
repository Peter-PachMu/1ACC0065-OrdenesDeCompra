package ml

import "math"

const NF = 24     // cantidad de variables (igual que en data_loader.go y ml_engine.go)
const Bins = 2000 // casilleros del histograma de puntajes 
const zMin = -10.0
const zMax = 10.0

// Columnas continuas que se estandarizan: x0 (log monto), x1 (largo descripción), x23 (tasa histórica)
var ColsContinuas = []int{0, 1, 23}


type Mensaje struct {
	Tipo    string    `json:"tipo"`
	W       []float64 `json:"w,omitempty"`        // pesos actuales del modelo
	Media   []float64 `json:"media,omitempty"`    // para estandarizar
	Desv    []float64 `json:"desv,omitempty"`     // para estandarizar
	PesoPos float64   `json:"peso_pos,omitempty"` // peso de las órdenes anuladas
	Split   string    `json:"split,omitempty"`    // "val" o "test"
}

// Respuesta: lo que el nodo le contesta al coordinador
type Respuesta struct {
	Error     string    `json:"error,omitempty"`
	N         int       `json:"n"`
	Positivos int       `json:"positivos"`
	Suma      []float64 `json:"suma,omitempty"`      // suma de cada variable continua
	SumaCuad  []float64 `json:"suma_cuad,omitempty"` // suma de cuadrados
	Grad      []float64 `json:"grad,omitempty"`      // gradiente parcial
	Peso      float64   `json:"peso"`
	Perdida   float64   `json:"perdida"`
	HistPos   []int     `json:"hist_pos,omitempty"` // histograma de puntajes de las anuladas
	HistNeg   []int     `json:"hist_neg,omitempty"` // histograma de puntajes de las no anuladas
}

func Sigmoide(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Estandariza (en el lugar) las columnas continuas de una fila
func Estandarizar(x *[NF]float64, media, desv []float64) {
	for k, j := range ColsContinuas {
		x[j] = (x[j] - media[k]) / desv[k]
	}
}

// Puntaje lineal z = w1*x1 + ... + b (el último peso es el sesgo)
func Puntaje(w []float64, x *[NF]float64) float64 {
	z := w[NF]
	for j := 0; j < NF; j++ {
		z += w[j] * x[j]
	}
	return z
}

// En qué casillero del histograma cae un puntaje
func Bin(z float64) int {
	if z < zMin {
		z = zMin
	}
	if z >= zMax {
		z = zMax - 0.0001
	}
	return int((z - zMin) / (zMax - zMin) * Bins)
}

type Metricas struct {
	AUCPR       float64 `json:"auc_pr"`
	Recall5     float64 `json:"recall_top5"`
	Precision5  float64 `json:"precision_top5"`
	Prevalencia float64 `json:"prevalencia"`
	Lift5       float64 `json:"lift_top5"`
	Ordenes     int     `json:"ordenes"`
	Anuladas    int     `json:"anuladas"`
}

// Calcula AUC-PR y los resultados del 5 % de mayor riesgo a partir de los histogramas
// que mandan los nodos (cada nodo cuenta cuántas anuladas y no anuladas hay en cada casillero).
func MetricasDeHistograma(pos, neg []int) Metricas {
	var P, N int
	for b := range pos {
		P += pos[b]
		N += neg[b]
	}
	m := Metricas{Ordenes: P + N, Anuladas: P}
	if P == 0 {
		return m
	}
	m.Prevalencia = float64(P) / float64(P+N)
	corte := (P + N) / 20 // el 5 % de las órdenes
	if corte < 1 {
		corte = 1
	}
	aciertos, total := 0, 0
	recallAnterior := 0.0
	tomado := false
	for b := Bins - 1; b >= 0; b-- { // de mayor a menor puntaje
		if pos[b]+neg[b] == 0 {
			continue
		}
		aciertos += pos[b]
		total += pos[b] + neg[b]
		precision := float64(aciertos) / float64(total)
		recall := float64(aciertos) / float64(P)
		m.AUCPR += (recall - recallAnterior) * precision
		recallAnterior = recall
		if !tomado && total >= corte {
			m.Recall5, m.Precision5, tomado = recall, precision, true
		}
	}
	m.Lift5 = m.Precision5 / m.Prevalencia
	return m
}
