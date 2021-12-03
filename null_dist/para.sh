#!/bin/bash


type="G F"

model="SBM ER WS"

for a in $type; do
    for b in $model; do
      for c in {1..100}; do
          echo "$a,$b,$c"
   done;
  done;
done;