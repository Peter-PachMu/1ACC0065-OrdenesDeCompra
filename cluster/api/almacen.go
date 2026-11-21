// almacen.go: todo lo relacionado con las bases de datos.
//   - MongoDB: guarda de forma permanente los entrenamientos, las predicciones y el estado de los nodos.
//   - Redis:   guarda respuestas temporales (caché) para no recalcular consultas repetidas.
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// ---------------------------------------------------------------------------
// Entidades (documentos) que se guardan en MongoDB
// ---------------------------------------------------------------------------

// Colección "entrenamientos": una corrida de entrenamiento del cluster
type Entrenamiento struct {
	ID             string    `bson:"_id" json:"id"`
	Inicio         time.Time `bson:"inicio" json:"inicio"`
	Fin            time.Time `bson:"fin" json:"fin"`
	Estado         string    `bson:"estado" json:"estado"` // en_curso, terminado, error...
	Nodos          int       `bson:"nodos" json:"nodos"`
	Iteraciones    int       `bson:"iteraciones" json:"iteraciones"`
	LR             float64   `bson:"lr" json:"lr"`
	Lambda         float64   `bson:"lambda" json:"lambda"`
	PesoPos        float64   `bson:"peso_pos" json:"peso_pos"`
	PerdidaFinal   float64   `bson:"perdida_final" json:"perdida_final"`
	AUCPRVal       float64   `bson:"auc_pr_val" json:"auc_pr_val"`
	AUCPRTest      float64   `bson:"auc_pr_test" json:"auc_pr_test"`
	Recall5Test    float64   `bson:"recall_top5_test" json:"recall_top5_test"`
	Precision5Test float64   `bson:"precision_top5_test" json:"precision_top5_test"`
	Lift5Test      float64   `bson:"lift_top5_test" json:"lift_top5_test"`
	Segundos       float64   `bson:"segundos" json:"segundos"`
	Pesos          []float64 `bson:"pesos" json:"pesos,omitempty"`
}

// Colección "predicciones": cada consulta que respondió la API
type Prediccion struct {
	Fecha        time.Time         `bson:"fecha" json:"fecha"`
	Entrada      EntradaPrediccion `bson:"entrada" json:"entrada"`
	Probabilidad float64           `bson:"probabilidad" json:"probabilidad"`
	Nivel        string            `bson:"nivel_riesgo" json:"nivel_riesgo"`
	LatenciaMs   float64           `bson:"latencia_ms" json:"latencia_ms"`
	DesdeCache   bool              `bson:"desde_cache" json:"desde_cache"`
}

// Colección "nodos": último estado conocido de cada nodo del cluster
type EstadoNodo struct {
	Direccion     string    `bson:"_id" json:"direccion"`
	Activo        bool      `bson:"activo" json:"activo"`
	LatenciaMs    float64   `bson:"latencia_ms" json:"latencia_ms"`
	MuestrasTrain int       `bson:"muestras_train" json:"muestras_train"`
	Revisado      time.Time `bson:"revisado" json:"revisado"`
}

var (
	colEntrenamientos *mongo.Collection
	colPredicciones   *mongo.Collection
	colNodos          *mongo.Collection
	rdb               *redis.Client
)

func contexto() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

// Conecta con MongoDB y Redis
func conectarBD() error {
	ctx, cancel := contexto()
	defer cancel()

	cliente, err := mongo.Connect(ctx, options.Client().ApplyURI(entorno("MONGO_URI", "mongodb://localhost:27017")))
	if err != nil {
		return err
	}
	if err := cliente.Ping(ctx, nil); err != nil {
		return err
	}
	bd := cliente.Database("compras")
	colEntrenamientos = bd.Collection("entrenamientos")
	colPredicciones = bd.Collection("predicciones")
	colNodos = bd.Collection("nodos")

	rdb = redis.NewClient(&redis.Options{Addr: entorno("REDIS_ADDR", "localhost:6379")})
	return rdb.Ping(ctx).Err()
}

// ---------------------------------------------------------------------------
// MongoDB
// ---------------------------------------------------------------------------

func guardarEntrenamiento(e Entrenamiento) {
	if colEntrenamientos == nil {
		return
	}
	ctx, cancel := contexto()
	defer cancel()
	_, err := colEntrenamientos.ReplaceOne(ctx, bson.M{"_id": e.ID}, e, options.Replace().SetUpsert(true))
	if err != nil {
		fmt.Println("error guardando el entrenamiento:", err)
	}
}

func listarEntrenamientos(n int64) ([]Entrenamiento, error) {
	ctx, cancel := contexto()
	defer cancel()
	opciones := options.Find().
		SetSort(bson.D{{Key: "inicio", Value: -1}}).
		SetLimit(n).
		SetProjection(bson.M{"pesos": 0})
	cursor, err := colEntrenamientos.Find(ctx, bson.M{}, opciones)
	if err != nil {
		return nil, err
	}
	salida := []Entrenamiento{}
	if err := cursor.All(ctx, &salida); err != nil {
		return nil, err
	}
	return salida, nil
}

func guardarPrediccion(p Prediccion) {
	if colPredicciones == nil {
		return
	}
	ctx, cancel := contexto()
	defer cancel()
	if _, err := colPredicciones.InsertOne(ctx, p); err != nil {
		fmt.Println("error guardando la predicción:", err)
	}
}

func listarPredicciones(n int64) ([]Prediccion, error) {
	ctx, cancel := contexto()
	defer cancel()
	opciones := options.Find().SetSort(bson.D{{Key: "fecha", Value: -1}}).SetLimit(n)
	cursor, err := colPredicciones.Find(ctx, bson.M{}, opciones)
	if err != nil {
		return nil, err
	}
	salida := []Prediccion{}
	if err := cursor.All(ctx, &salida); err != nil {
		return nil, err
	}
	return salida, nil
}

func guardarNodo(n EstadoNodo) {
	if colNodos == nil {
		return
	}
	ctx, cancel := contexto()
	defer cancel()
	_, err := colNodos.ReplaceOne(ctx, bson.M{"_id": n.Direccion}, n, options.Replace().SetUpsert(true))
	if err != nil {
		fmt.Println("error guardando el nodo:", err)
	}
}

// ---------------------------------------------------------------------------
// Redis (caché). Si Redis falla, la API sigue funcionando sin caché.
// ---------------------------------------------------------------------------

func cacheGet(clave string) (string, bool) {
	if rdb == nil {
		return "", false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	valor, err := rdb.Get(ctx, clave).Result()
	if err != nil { // no existe (redis.Nil) o hubo un error
		return "", false
	}
	return valor, true
}

func cacheSet(clave, valor string, duracion time.Duration) {
	if rdb == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	rdb.Set(ctx, clave, valor, duracion)
}
