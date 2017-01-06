"""https://gist.github.com/andreasvc/7507135#file-tigersplit-py"""
""" The train-test split described in Hall & Nivre (2008),
Parsing Discontinuous Phrase Structure with Grammatical Functions.

Corpus is divided in Sections 0-9, where sentence i is allocated to section i mod 10.
For development train on sections 2-9; evaluate on section 1.
For final evaluation (test) train on sections 1-9; evaluate on section 0.
"""
import io
import os
from discodop.treebank import NegraCorpusReader

#corpus = NegraCorpusReader('tiger/corpus', 'tiger_release_aug07.export',
		#encoding='iso-8859-1')

corpus = NegraCorpusReader('tiger21/corpus/tiger_release_aug07.export',
		encoding='iso-8859-1')

#os.mkdir('tiger-split/')
io.open('tiger21/tigertraindev.export', 'w', encoding='utf8').writelines(
		a for n, a in enumerate(corpus.blocks().values(), 1)
		if n % 10 > 1)
io.open('tiger21/tigerdev.export', 'w', encoding='utf8').writelines(
		a for n, a in enumerate(corpus.blocks().values(), 1)
		if n % 10 == 1)

io.open('tiger21/tigertraintest.export', 'w', encoding='utf8').writelines(
		a for n, a in enumerate(corpus.blocks().values(), 1)
		if n % 10 != 0)
io.open('tiger21/tigertest.export', 'w', encoding='utf8').writelines(
		a for n, a in enumerate(corpus.blocks().values(), 1)
		if n % 10 == 0)
