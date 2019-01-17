#!/bin/bash

git pull origin master
git add *.txt *.csv *.pt 
git commit -m "Add records and models"
git push origin master 
