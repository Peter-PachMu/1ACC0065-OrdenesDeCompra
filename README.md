# 1ACC0065 - Predicción de Riesgo y Anulación en Compras Públicas (Tema 4)

Este proyecto implementa una solución de **Programación Concurrente y Distribuida en Go** para el análisis masivo de **Órdenes de Compra y Servicio del Estado Peruano** (dataset del OSCE / SEACE con más de 4.0 millones de registros).

Alumno: Peter Smith Pacherres Muñoz

Codigo: U202423854

---

## 🎯 Impacto Social y Pregunta de Investigación

El sistema busca responder a la siguiente pregunta de impacto social:

> **¿Qué órdenes de compra y de servicio del Estado, especialmente de entidades de salud y servicios básicos, tienen alta probabilidad de ser anuladas después de su emisión, para que las entidades intervengan a tiempo y se eviten retrasos en la atención al ciudadano, y para que los órganos de control prioricen su revisión?**

---

## 📊 Fuente de Datos

Los datos provienen del portal oficial del **SEACE / OSCE (Organismo Supervisor de las Contrataciones del Estado)**:
* **Enlace a los Datasets Oficiales:** [Dataset de Órdenes de Compra y Servicio - SEACE](https://bi.seace.gob.pe/pentaho/api/repos/%3Apublic%3Aportal%3Adataset.html/content?userid=public&password=key&pagina=ordenes)

---

## 🛠️ Arquitectura del Sistema (Go Concurrente y Distribuido)

El proyecto está compuesto por dos módulos principales desarrollados íntegramente en lenguaje **Go**:

1. **`data_loader.go` (Parte A — Ingestión y Limpieza Concurrente):**
   * Implementa un pipeline concurrente en 3 etapas usando `goroutines`, `channels` y `sync.WaitGroup`.
   * Parsea archivos masivos en formato `.xlsx` procesando múltiples archivos e hilos de limpieza en paralelo.
   * Realiza la filtración de duplicados, detección de anulaciones masivas administrativas y la generación de **23 características numéricas (features)** para el modelo.
   * Particiona los datos cronológicamente en conjuntos de **Train**, **Validación** y **Test**.

2. **`ml_engine.go` (Parte B — Motor ML Distribuido):**
   * Implementa el algoritmo de **Regresión Logística Distribuida** con **Descenso de Gradiente**.
   * Aplica un patrón de computación distribuida mediante \\(K\\) goroutines (nodos de cómputo) que calculan gradientes parciales sobre sus bloques de datos y se comunican con el coordinador mediante `channels`.
   * Incluye manejo de desbalance de clases (`pesoPos`), regularización L2 y un modo de verificación numérica (`-verificar`).

---

## 📁 Estructura del Repositorio

```text
├── data_loader.go       # Cargador y limpiador concurrente en Go
├── ml_engine.go         # Motor de Machine Learning distribuido en Go
├── go.mod               # Módulo de dependencias en Go
├── go.sum               # Hash de comprobación de librerías
├── .gitignore           # Exclusión de datos pesados y temporales
└── README.md            # Documentación del proyecto
```

##  Instrucciones de Ejecución

## Prerrequisitos

- Tener instalado Go 1.20+.
- Instalar la librería externa para lectura de archivos Excel:

```bash
go get github.com/xuri/excelize/v2
```

## Paso 1: Preparar los Datos

Crea una carpeta llamada datos/ en la raíz del proyecto y coloca ahí los archivos mensuales en formato .xlsx descargados del portal del SEACE:

```text
/1ACC0065-OrdenesDeCompra
  └── datos/
      ├── ORDENESCOMPRAAGOSTO2025_0.xlsx
      ├── ORDENESCOMPRASETIEMBRE2025_0.xlsx
      └── ...
```

## Paso 2: Ejecutar el Cargador y Limpiador Concurrente

Ejecuta el siguiente comando en tu terminal:

```bash
go run data_loader.go -in ./datos -out dataset_limpio.csv
```

Generará el archivo dataset_limpio.csv listo para el modelo ML.

## Paso 3: Ejecutar el Motor de Machine Learning Distribuido

### Entrenamiento normal con 4 nodos (goroutines)

```bash
go run ml_engine.go -datos dataset_limpio.csv -nodos 4
```

### Modo Verificación (Comprueba la consistencia distribuida vs secuencial)

```bash
go run ml_engine.go -datos dataset_limpio.csv -nodos 4 -verificar
```