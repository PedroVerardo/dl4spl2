# Como rodar o notebook?

## 0 Verifique as especificaçõe do seu computador(se possuir GPU NVIDIA)
- O pytorch é sensível a mudança de versão do CUDA

        nvidia-smi
> A informação estará no canto superior direito CUDA: <versão>

- Rodar o cmando especificado pelo pytorch (https://pytorch.org/)


## 1 Baixar dependencias (ler tudo antes de rodar)
- Crie uma venv local

        pip install -r requirements.txt
- Ppcionalmente crie uma venv. Isso manterá suas dependencias organizadas e centralizadas. você pode fazer isso de 2 formas
        
        python -m venv <nome da sua venv>
        conda create --name <my-env> 

A venv do anaconda é interessante nesse caso, já que você pode centralizar o pytorch. O pytorch normalmente quebra outras bibliotecas devido ao numero de dependencias e principalmente pela versão do numpy, que normalmente varia bastante.

## 2 Tente rodar da forma que o notebook para coferir se tudo esta reodando com os conformes
- Apenas use o run all no topo do notebook

## 3 Atualize as informações do notebook
Existe 3 lugares no código onde se deve alterar para obter uma melhor performance eles são sinalizados por <numero>+ . Leia e mude o valor<br>
Ex: 1+ 2+ 3+ 