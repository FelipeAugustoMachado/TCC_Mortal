# TCC_Mortal

# Introdução
Esse repositório possui todos os arquivos utilizados na elaboração da parte de pré-processamento do nosso TCC.
### Autores:
Felipe Augusto de Moraes Machado - felipe.augusto.machado@usp.br
Andrés Eduardo Marchetti Colognesi - andres.colognesi@usp.br

# Estrutura
## Pré-processamento
Os arquivos de pré-processamento estão na pasta "Dados Internet". Nele, há uma pasta chamada CODE que apresenta diversos arquivos dentro:
* 2 arquivos .py, onde estão as classes e funções desenvolvidas
* 3 scripts .ipynb, onde estão a pipeline do pré-processamento e modelagem
* 3 arquivos .pkl, que são os modelos desenvolvidos.

## Simulação do Controle
As classes que implementam o controle de baixo nível (LLC) e de alto nível (HLC) para simulação do controle estão colocadas no arquivo "controle.py", na pasta "/Simu" deste repositório.

## Análises Adicionais da Modelagem de Controle 
Na pasta "/Análises Apêndice" estão os códigos em Matlab desenvolvidos para analisar os fatores dinâmicos desconsiderados no modelo nominal do projeto (como explicitado na Monografia).

## Simulação do sistema integrado
A simulação do sistema como um todo é realizada por meio dos arquivos contidos na pasta "/Simu", onde o arquivo "integracao_c.py" é o arquivo principal da simulação.
