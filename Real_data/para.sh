#!/bin/bash


type="reddit twitch"

for a in $type; do
  for b in {1..2000}; do
        echo "$a,$b"
  done;
done;