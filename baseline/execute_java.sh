#!/bin/bash
cd /home/hltcoe/estengel/miso_decomp/baseline

java -cp "*" -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse -output.prettyPrint true -outputFormat conllu -file $1 -outputDirectory $2 -sent    ences newline -tokenize.whitespace -ssplit.eolonly
