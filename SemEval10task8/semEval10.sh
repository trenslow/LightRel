#!/bin/bash

s=4
c=0.1
e=0.3
out_name="s${s//.}c${c//.}e${e//.}"
mainDir=$(pwd)
featDir=${mainDir}"/features/"
modelDir=${mainDir}"/models/"
libLinDir="/home/tyler/liblinear-2.11"
semEvalScorerDir=${mainDir}"/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2"

echo "---running feature extraction---"
python3 featureExtraction10.py
ret=$?
if [ ${ret} == 1 ]; then
     exit 1
fi
echo "---converting sentences to vectors---"
python3 sent2vec10.py

cd ${libLinDir}
echo "---training LibLinear model---"
./train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
echo "---predicting on test set---"
./predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}${out_name}"_predictions.txt"

cd ${mainDir}
echo "---adding labels to LibLinear output---"
python3 addLabels10.py ${s} ${c} ${e}
echo "---scoring model---"
perl ${semEvalScorerDir}"/semeval2010_task8_scorer-v1.2.pl" ${modelDir}${out_name}"_predictions_with_labels.txt" answer_key10.txt