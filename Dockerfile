ARG GO_VERSION=1.27.1
FROM golang:${GO_VERSION} AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cluster ./cluster
ARG SERVICIO
RUN CGO_ENABLED=0 go build -o /app ./cluster/${SERVICIO}

FROM alpine:3.20
COPY --from=build /app /app
CMD ["/app"]
