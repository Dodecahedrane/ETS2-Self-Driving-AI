#!/bin/bash

for key in $(redis-cli -p 6379 keys \*);
  do echo "Key : '$key'" 
     redis-cli -p 6379 GET $key;
done