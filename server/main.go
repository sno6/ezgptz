package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

const (
	dataFilePath = "/Users/farleyschaefer/Documents/projects/newco/engine/data/training.json"

	trainPath = "/Users/farleyschaefer/Documents/projects/newco/engine/train.py"
	evalPath = "/Users/farleyschaefer/Documents/projects/newco/engine/eval.py"
)


type EventsList struct {
	Events []string `json:"events"`
}

func handleTrain(ctx context.Context, w http.ResponseWriter, events EventsList) {
	log.Printf("Request to train %d events\n", len(events.Events))

	b, err := json.Marshal(events.Events)
	if err != nil {
		log.Println(err.Error())
		return
	}

	// Write events into training data file.
	f, err := os.Create(dataFilePath)
	if err != nil {
		log.Println(err.Error())
		return
	}
	defer func() {
		if err := f.Close(); err != nil {
			log.Println(err.Error())
		}
	}()

	if _, err := f.Write(b); err != nil {
		log.Println(err.Error())
		return
	}

	cmd := exec.CommandContext(ctx, "python", trainPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Println(err.Error())
	}

	log.Println("trained")
}

func handleEval(w http.ResponseWriter, seed string) {
	log.Printf("Eval mode, seed: %s\n", seed)

	ctx, done := context.WithTimeout(context.Background(), time.Second * 10)
	defer done()

	buf := &bytes.Buffer{}

	cmd := exec.CommandContext(ctx, "python", evalPath, "-s", seed)
	cmd.Stdout = buf
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Println(err.Error())
		return
	}

	if _, err := w.Write(buf.Bytes()); err != nil {
		log.Println(err.Error())
	}
}


func main() {
	var mu sync.Mutex
	ctx, done := context.WithCancel(context.Background())

	mux := http.NewServeMux()
	mux.HandleFunc("/train", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		// Kill any existing training scripts and replace the context.
		done()
		ctx, done = context.WithCancel(context.Background())

		var events EventsList
		if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
			log.Println(err.Error())
			return
		}
		if len(events.Events) == 0 {
			log.Println("No events in request.")
			return
		}

		go handleTrain(ctx, w, events)
	})

	mux.HandleFunc("/eval", func(w http.ResponseWriter, r *http.Request) {
		type EvalReq struct {
			Seed string `json:"seed"`
		}

		var req EvalReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Println(err.Error())
			return
		}

		handleEval(w, req.Seed)
	})

	log.Fatal(http.ListenAndServe(":8080", mux))
}
