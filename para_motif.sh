#!/bin/bash

type="H1"

for a in $type; do
  for b in {1..100}; do
        echo "$a,$b"
  done;
done;