from scipy.stats.stats import pearsonr


def Correlacion(dictionaryAction,accionesAgent):
    for i in range(0,len(dictionaryAction['Move'])):
        corr = dictionaryAction['Move'][i]
        if len(corr) == 0:
            continue
        nueva = accionesAgent[0:len(corr)]
        correlacion = pearsonr(corr, nueva)
        print(correlacion)