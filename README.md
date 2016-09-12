# QV_work
Work at QV

Some descriptive statistics about my project

def lista_mod(columna):
    l=[]
    for i in columna:
        if i not in l:
            l.append(i)
    return l

trata las columnas numericas y otras

def vectorisation_all(df_columna,lista_mod):

    if type(df_columna[0])!=numpy.int64:

        dico={}
        for i in range(len(lista_mod)):
            dico[lista_mod[i]]=i
    
        num_dom=[]
        for i in df_columna:
            num_dom.append(dico[i])
            
    else:
        dico={}
        for i in lista_mod:
            dico[i]=i
        
        num_dom=[]
        for i in df_columna:
            num_dom.append(dico[i])
        
    
    return(num_dom, dico)
