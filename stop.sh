#!/bin/bash

while (sleep 1);
do
    ps -aux | grep python | grep -v python3 | grep $1 | awk '{print $2}' | xargs kill;
done

