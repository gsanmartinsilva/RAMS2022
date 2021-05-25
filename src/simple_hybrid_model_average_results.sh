#!/bin/sh
count=1
totalRun=10

while [ $count -le $totalRun ]
do
  seed=$(sed -n "$count p" seeds.txt)
  printf "Count has a value of $count\n"
  printf "Seed: $seed\n"
  guild run simple_hybrid_model:train seed=$seed epochs=1 -y
  count=$(($count+1))
done