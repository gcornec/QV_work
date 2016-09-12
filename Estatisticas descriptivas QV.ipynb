{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Populating the interactive namespace from numpy and matplotlib\n"
     ]
    }
   ],
   "source": [
    "%pylab inline\n",
    "import pyensae\n",
    "import pandas, urllib.request\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import time\n",
    "from sklearn import preprocessing\n",
    "from sklearn import neural_network\n",
    "from sklearn import linear_model\n",
    "from sklearn import ensemble\n",
    "from sklearn import metrics\n",
    "from sklearn import neighbors\n",
    "from sklearn import svm\n",
    "from sklearn.metrics import roc_curve, auc\n",
    "from sklearn.decomposition import PCA\n",
    "from pydoc import help\n",
    "from scipy.stats.stats import pearsonr, spearmanr\n",
    "from random import randrange"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "t3=time.clock()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Importacion de la base"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/IPython/core/interactiveshell.py:2705: DtypeWarning: Columns (9) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  interactivity=interactivity, compiler=compiler, result=result)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Client</th>\n",
       "      <th>from address</th>\n",
       "      <th>from domain</th>\n",
       "      <th>destination user</th>\n",
       "      <th>has only words</th>\n",
       "      <th>has special chars</th>\n",
       "      <th>destination domain</th>\n",
       "      <th>attachments</th>\n",
       "      <th>has attachments</th>\n",
       "      <th>...</th>\n",
       "      <th>status_name</th>\n",
       "      <th>created</th>\n",
       "      <th>bounce detail</th>\n",
       "      <th>num_dominio</th>\n",
       "      <th>num_client</th>\n",
       "      <th>num_words</th>\n",
       "      <th>num_chars</th>\n",
       "      <th>attach</th>\n",
       "      <th>num_from</th>\n",
       "      <th>bounce</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>destination</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>jobediente@gfaralon.com</th>\n",
       "      <td>0</td>\n",
       "      <td>BAC</td>\n",
       "      <td>credomatic-informa@pa.credomatic.com</td>\n",
       "      <td>pa.credomatic.com</td>\n",
       "      <td>jobediente</td>\n",
       "      <td>Y</td>\n",
       "      <td>N</td>\n",
       "      <td>gfaralon.com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>Invalid domain</td>\n",
       "      <td>08/07/2016 0:00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2149</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>luisanaisa@hotmail.com</th>\n",
       "      <td>1</td>\n",
       "      <td>BAC</td>\n",
       "      <td>credomatic-informa@pa.credomatic.com</td>\n",
       "      <td>pa.credomatic.com</td>\n",
       "      <td>luisanaisa</td>\n",
       "      <td>Y</td>\n",
       "      <td>N</td>\n",
       "      <td>hotmail.com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>Delivered</td>\n",
       "      <td>08/06/2016 23:59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2662</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>cmonte30@hotmail.com</th>\n",
       "      <td>2</td>\n",
       "      <td>BAC</td>\n",
       "      <td>credomatic-informa@pa.credomatic.com</td>\n",
       "      <td>pa.credomatic.com</td>\n",
       "      <td>cmonte30</td>\n",
       "      <td>N</td>\n",
       "      <td>N</td>\n",
       "      <td>hotmail.com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>Viewed</td>\n",
       "      <td>08/06/2016 23:59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2662</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>roderick@mcgowensa.com</th>\n",
       "      <td>3</td>\n",
       "      <td>BAC</td>\n",
       "      <td>credomatic-informa@pa.credomatic.com</td>\n",
       "      <td>pa.credomatic.com</td>\n",
       "      <td>roderick</td>\n",
       "      <td>Y</td>\n",
       "      <td>N</td>\n",
       "      <td>mcgowensa.com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>Viewed</td>\n",
       "      <td>08/06/2016 23:59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3457</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>manuelguevara@gmail.com</th>\n",
       "      <td>4</td>\n",
       "      <td>BAC</td>\n",
       "      <td>credomatic-informa@pa.credomatic.com</td>\n",
       "      <td>pa.credomatic.com</td>\n",
       "      <td>manuelguevara</td>\n",
       "      <td>Y</td>\n",
       "      <td>N</td>\n",
       "      <td>gmail.com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>...</td>\n",
       "      <td>Delivered</td>\n",
       "      <td>08/06/2016 23:59</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2209</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                         Unnamed: 0 Client  \\\n",
       "destination                                  \n",
       "jobediente@gfaralon.com           0    BAC   \n",
       "luisanaisa@hotmail.com            1    BAC   \n",
       "cmonte30@hotmail.com              2    BAC   \n",
       "roderick@mcgowensa.com            3    BAC   \n",
       "manuelguevara@gmail.com           4    BAC   \n",
       "\n",
       "                                                 from address  \\\n",
       "destination                                                     \n",
       "jobediente@gfaralon.com  credomatic-informa@pa.credomatic.com   \n",
       "luisanaisa@hotmail.com   credomatic-informa@pa.credomatic.com   \n",
       "cmonte30@hotmail.com     credomatic-informa@pa.credomatic.com   \n",
       "roderick@mcgowensa.com   credomatic-informa@pa.credomatic.com   \n",
       "manuelguevara@gmail.com  credomatic-informa@pa.credomatic.com   \n",
       "\n",
       "                               from domain destination user has only words  \\\n",
       "destination                                                                  \n",
       "jobediente@gfaralon.com  pa.credomatic.com       jobediente              Y   \n",
       "luisanaisa@hotmail.com   pa.credomatic.com       luisanaisa              Y   \n",
       "cmonte30@hotmail.com     pa.credomatic.com         cmonte30              N   \n",
       "roderick@mcgowensa.com   pa.credomatic.com         roderick              Y   \n",
       "manuelguevara@gmail.com  pa.credomatic.com    manuelguevara              Y   \n",
       "\n",
       "                        has special chars destination domain attachments  \\\n",
       "destination                                                                \n",
       "jobediente@gfaralon.com                 N       gfaralon.com         NaN   \n",
       "luisanaisa@hotmail.com                  N        hotmail.com         NaN   \n",
       "cmonte30@hotmail.com                    N        hotmail.com         NaN   \n",
       "roderick@mcgowensa.com                  N      mcgowensa.com         NaN   \n",
       "manuelguevara@gmail.com                 N          gmail.com         NaN   \n",
       "\n",
       "                        has attachments   ...       status_name  \\\n",
       "destination                               ...                     \n",
       "jobediente@gfaralon.com               N   ...    Invalid domain   \n",
       "luisanaisa@hotmail.com                N   ...         Delivered   \n",
       "cmonte30@hotmail.com                  N   ...            Viewed   \n",
       "roderick@mcgowensa.com                N   ...            Viewed   \n",
       "manuelguevara@gmail.com               N   ...         Delivered   \n",
       "\n",
       "                                  created bounce detail num_dominio  \\\n",
       "destination                                                           \n",
       "jobediente@gfaralon.com   08/07/2016 0:00           NaN        2149   \n",
       "luisanaisa@hotmail.com   08/06/2016 23:59           NaN        2662   \n",
       "cmonte30@hotmail.com     08/06/2016 23:59           NaN        2662   \n",
       "roderick@mcgowensa.com   08/06/2016 23:59           NaN        3457   \n",
       "manuelguevara@gmail.com  08/06/2016 23:59           NaN        2209   \n",
       "\n",
       "                         num_client  num_words  num_chars  attach  num_from  \\\n",
       "destination                                                                   \n",
       "jobediente@gfaralon.com           0          1          0       0         5   \n",
       "luisanaisa@hotmail.com            0          1          0       0         5   \n",
       "cmonte30@hotmail.com              0          0          0       0         5   \n",
       "roderick@mcgowensa.com            0          1          0       0         5   \n",
       "manuelguevara@gmail.com           0          1          0       0         5   \n",
       "\n",
       "                         bounce  \n",
       "destination                      \n",
       "jobediente@gfaralon.com       1  \n",
       "luisanaisa@hotmail.com        0  \n",
       "cmonte30@hotmail.com          0  \n",
       "roderick@mcgowensa.com        0  \n",
       "manuelguevara@gmail.com       0  \n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Importamos la base completa\n",
    "base_c=r'/Users/gillescornec/Desktop/Bases_QV/base_complete.csv'\n",
    "df_base_c = pandas.read_csv(base_c, sep=';')\n",
    "df_base=df_base_c.set_index(\"destination\")\n",
    "df_base.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Unnamed: 0', 'Client', 'from address', 'from domain',\n",
       "       'destination user', 'has only words', 'has special chars',\n",
       "       'destination domain', 'attachments', 'has attachments', 'status_id',\n",
       "       'status_name', 'created', 'bounce detail', 'num_dominio', 'num_client',\n",
       "       'num_words', 'num_chars', 'attach', 'num_from', 'bounce'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_base.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(121903, 21)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#base de 21 variables y de 121903 observaciones.\n",
    "shape(df_base)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Estudio de las correlaciones"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Debemos vectorizar las variables y normalizarles antes de hacer test de correlacion."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def lista_mod(columna):\n",
    "    l=[]\n",
    "    for i in columna:\n",
    "        if i not in l:\n",
    "            l.append(i)\n",
    "    return l"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#trata las columnas numericas y otras\n",
    "def vectorisation_all(df_columna,lista_mod):\n",
    "\n",
    "    if type(df_columna[0])!=numpy.int64:\n",
    "\n",
    "        dico={}\n",
    "        for i in range(len(lista_mod)):\n",
    "            dico[lista_mod[i]]=i\n",
    "    \n",
    "        num_dom=[]\n",
    "        for i in df_columna:\n",
    "            num_dom.append(dico[i])\n",
    "            \n",
    "    else:\n",
    "        dico={}\n",
    "        for i in lista_mod:\n",
    "            dico[i]=i\n",
    "        \n",
    "        num_dom=[]\n",
    "        for i in df_columna:\n",
    "            num_dom.append(dico[i])\n",
    "        \n",
    "    \n",
    "    return(num_dom, dico)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def df_vectorisé(base,liste_variable):\n",
    "    \n",
    "    #var_name=[\"destination domain\",\"Client\",\"from domain\",\"has attachments\", \"has only words\", \"has special chars\"]\n",
    "    l_vect=[]\n",
    "    var_dic=[]\n",
    "    \n",
    "    for i in liste_variable:\n",
    "        l_vect.append(pandas.DataFrame(vectorisation_all(base[i],lista_mod(base[i]))[0],columns=[i]))\n",
    "        var_dic.append(vectorisation_all(base[i],lista_mod(base[i]))[1])\n",
    "    #df_vect=pandas.concat(l_vect, axis=1)\n",
    "    \n",
    "    return (pandas.concat(l_vect, axis=1), var_dic)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.8822\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>destination domain</th>\n",
       "      <th>Client</th>\n",
       "      <th>from domain</th>\n",
       "      <th>has attachments</th>\n",
       "      <th>has only words</th>\n",
       "      <th>has special chars</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   destination domain  Client  from domain  has attachments  has only words  \\\n",
       "0                   0       0            0                0               0   \n",
       "1                   1       0            0                0               0   \n",
       "2                   1       0            0                0               1   \n",
       "3                   2       0            0                0               0   \n",
       "4                   3       0            0                0               0   \n",
       "\n",
       "   has special chars  \n",
       "0                  0  \n",
       "1                  0  \n",
       "2                  0  \n",
       "3                  0  \n",
       "4                  0  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t1=time.clock()\n",
    "var_name=[\"destination domain\",\"Client\",\"from domain\",\"has attachments\", \"has only words\", \"has special chars\"]\n",
    "df_vect=df_vectorisé(df_base,var_name)[0]\n",
    "t2=time.clock()\n",
    "print(t2-t1)\n",
    "df_vect.head()    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'N': 0, 'Y': 1}"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_vectorisé(df_base,var_name)[1][3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#solo funciona con dataframes.\n",
    "def correlacion_inputs_target(inputs,target):\n",
    "    t=pandas.DataFrame([{\"Input\":1 ,\"Target\":2,\"correlation\":4,\"p_value\":5 }])\n",
    "    l=[]\n",
    "    l.append(t)\n",
    "    for i in range(len(inputs.columns)):\n",
    "        a=l[i].append([{\"Input\":inputs.columns[i] ,\"Target\":'Target' ,\"correlation\":spearmanr(inputs[inputs.columns[i]],target)[0],\"p_value\":spearmanr(inputs[inputs.columns[i]],target)[1]}])\n",
    "        l.append(a)\n",
    "    \n",
    "    b=l[i+1].set_index(\"Target\")\n",
    "    c=b.drop([2])\n",
    "    return c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Input</th>\n",
       "      <th>correlation</th>\n",
       "      <th>p_value</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>destination domain</td>\n",
       "      <td>0.061045</td>\n",
       "      <td>5.576851e-101</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>Client</td>\n",
       "      <td>0.429016</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>from domain</td>\n",
       "      <td>0.423837</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>has attachments</td>\n",
       "      <td>-0.022557</td>\n",
       "      <td>3.362405e-15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>has only words</td>\n",
       "      <td>-0.048308</td>\n",
       "      <td>6.736180e-64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Target</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>-0.011214</td>\n",
       "      <td>9.023224e-05</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Input  correlation        p_value\n",
       "Target                                                \n",
       "Target  destination domain     0.061045  5.576851e-101\n",
       "Target              Client     0.429016   0.000000e+00\n",
       "Target         from domain     0.423837   0.000000e+00\n",
       "Target     has attachments    -0.022557   3.362405e-15\n",
       "Target      has only words    -0.048308   6.736180e-64\n",
       "Target   has special chars    -0.011214   9.023224e-05"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "correlacion_inputs_target(df_vect,df_base[\"bounce\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#correlacion entre los diff inputs.\n",
    "def correlacion_inputs(inputs):\n",
    "    t=pandas.DataFrame([{\"Input 1\":1 ,\"Input 2\":2,\"correlation\":4,\"p_value\":5 }])\n",
    "    l=[]\n",
    "    p=0\n",
    "    l.append(t)\n",
    "    for i in range(len(inputs.columns)):\n",
    "        for j in range(i+1,len(inputs.columns)):\n",
    "            #print(i)\n",
    "            #print(j)\n",
    "            a=l[p].append([{\"Input 1\":inputs.columns[i] ,\"Input 2\": inputs.columns[j],\"correlation\":spearmanr(inputs[inputs.columns[i]],inputs[inputs.columns[j]])[0],\"p_value\":spearmanr(inputs[inputs.columns[i]],inputs[inputs.columns[j]])[1]}])\n",
    "            l.append(a)\n",
    "            #print(spearmanr(base_inputs[base_inputs.columns[i]],base_inputs[base_inputs.columns[j]]))\n",
    "            #print(l[p])\n",
    "            p=p+1\n",
    "            #print(p)\n",
    "            \n",
    "    b=l[p].set_index(\"Input 1\")\n",
    "    c=b.drop([1])\n",
    "\n",
    "    return c"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Input 2</th>\n",
       "      <th>correlation</th>\n",
       "      <th>p_value</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Input 1</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>destination domain</th>\n",
       "      <td>Client</td>\n",
       "      <td>0.004777</td>\n",
       "      <td>9.534089e-02</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>destination domain</th>\n",
       "      <td>from domain</td>\n",
       "      <td>0.007600</td>\n",
       "      <td>7.966114e-03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>destination domain</th>\n",
       "      <td>has attachments</td>\n",
       "      <td>0.225311</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>destination domain</th>\n",
       "      <td>has only words</td>\n",
       "      <td>-0.280431</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>destination domain</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>-0.056999</td>\n",
       "      <td>2.892645e-88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Client</th>\n",
       "      <td>from domain</td>\n",
       "      <td>0.982523</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Client</th>\n",
       "      <td>has attachments</td>\n",
       "      <td>0.012844</td>\n",
       "      <td>7.306285e-06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Client</th>\n",
       "      <td>has only words</td>\n",
       "      <td>-0.033039</td>\n",
       "      <td>8.430172e-31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Client</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>-0.018727</td>\n",
       "      <td>6.197416e-11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>from domain</th>\n",
       "      <td>has attachments</td>\n",
       "      <td>0.012702</td>\n",
       "      <td>9.202738e-06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>from domain</th>\n",
       "      <td>has only words</td>\n",
       "      <td>-0.037105</td>\n",
       "      <td>2.079756e-38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>from domain</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>-0.019384</td>\n",
       "      <td>1.300967e-11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>has attachments</th>\n",
       "      <td>has only words</td>\n",
       "      <td>-0.059590</td>\n",
       "      <td>2.630236e-96</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>has attachments</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>0.034854</td>\n",
       "      <td>4.341448e-34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>has only words</th>\n",
       "      <td>has special chars</td>\n",
       "      <td>0.545594</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              Input 2  correlation       p_value\n",
       "Input 1                                                         \n",
       "destination domain             Client     0.004777  9.534089e-02\n",
       "destination domain        from domain     0.007600  7.966114e-03\n",
       "destination domain    has attachments     0.225311  0.000000e+00\n",
       "destination domain     has only words    -0.280431  0.000000e+00\n",
       "destination domain  has special chars    -0.056999  2.892645e-88\n",
       "Client                    from domain     0.982523  0.000000e+00\n",
       "Client                has attachments     0.012844  7.306285e-06\n",
       "Client                 has only words    -0.033039  8.430172e-31\n",
       "Client              has special chars    -0.018727  6.197416e-11\n",
       "from domain           has attachments     0.012702  9.202738e-06\n",
       "from domain            has only words    -0.037105  2.079756e-38\n",
       "from domain         has special chars    -0.019384  1.300967e-11\n",
       "has attachments        has only words    -0.059590  2.630236e-96\n",
       "has attachments     has special chars     0.034854  4.341448e-34\n",
       "has only words      has special chars     0.545594  0.000000e+00"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "correlacion_inputs(df_vect)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Estudio de los clientes: quienes son los que generan lo mas bounces?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#transformamos el df en array para calcular mas facilmente.\n",
    "np_bounce_all=df_base[\"bounce\"].as_matrix()\n",
    "np_client_all=df_base[\"Client\"].as_matrix()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['BAC', 'Banesco', 'Mapfre', 'Ricardo Perez']"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lista_mod(np_client_all)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def enumeration(columna,lista_modalidades):\n",
    "\n",
    "    dico={}\n",
    "    for i in range(len(lista_modalidades)):\n",
    "        dico[lista_modalidades[i]]=0\n",
    "        \n",
    "    for i in lista_modalidades:\n",
    "        #print(i)\n",
    "        for j in columna:\n",
    "            #print(j)\n",
    "            if i==j:\n",
    "                dico[i]=dico[i]+1\n",
    "    \n",
    "    return(dico)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def bounce_por_client(df_base):\n",
    "    \n",
    "    BDCC=[]\n",
    "    \n",
    "    np_client_all=df_base[\"Client\"].as_matrix()\n",
    "    \n",
    "    df_bounce=df_base[df_base[\"bounce\"]==1]\n",
    "    np_client_bounce=df_bounce[\"Client\"].as_matrix()\n",
    "    \n",
    "    dic_client_all=enumeration(np_client_all,lista_mod(np_client_all))\n",
    "    dic_client_bounce=enumeration(np_client_bounce,lista_mod(np_client_bounce))\n",
    "    \n",
    "    #print(dic_client_all)\n",
    "    #print(dic_client_bounce)\n",
    "    \n",
    "    for i in lista_mod(np_client_bounce):\n",
    "        BDCC.append(dic_client_bounce[i]/dic_client_all[i])\n",
    "        \n",
    "    return(BDCC)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.09660092993426327,\n",
       " 0.20051004144086707,\n",
       " 0.29273076923076924,\n",
       " 0.609215203122615]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bounce_por_client(df_base)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x1026d1fd0>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x10a74cb00>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.bar([0.5+i for i in range(len(bounce_por_client(df_base)))],bounce_por_client(df_base))\n",
    "plt.xticks([1+i for i in range(len(bounce_por_client(df_base)))], lista_mod(np_client_all))\n",
    "plt.title(\"% bounces para cada cliente\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['gfaralon.com', 'hotmail.com', 'hotmail.com', ..., 'hotmail.com',\n",
       "       'hotmail.com', 'hotmail.com'], dtype=object)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np_dominio=df_base[\"destination domain\"].as_matrix()\n",
    "np_dominio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Mejor: identificar los destinatario dominio que generan lo mas bounce (/frecuencia de aparencia de estos dominio)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###  Estudio de los dominios: cuales son los que generan los mas bounce?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Puede ser generalizado en bounce por modalidades de algun variable precisando la variable\n",
    "def bounce_por_dominios(df_base):\n",
    "    \n",
    "    BDCD={}\n",
    "    l=[]\n",
    "    np_dom_all=df_base[\"destination domain\"].as_matrix()\n",
    "    \n",
    "    df_bounce=df_base[df_base[\"bounce\"]==1]\n",
    "    np_dom_bounce=df_bounce[\"destination domain\"].as_matrix()\n",
    "    \n",
    "    dic_dom_all=enumeration(np_dom_all,lista_mod(np_dom_all))\n",
    "    dic_dom_bounce=enumeration(np_dom_bounce,lista_mod(np_dom_bounce))\n",
    "    \n",
    "    #print(dic_client_all)\n",
    "    #print(dic_client_bounce)\n",
    "    \n",
    "    for i in lista_mod(np_dom_bounce):\n",
    "        #no queremos los email adress sola \n",
    "        if dic_dom_bounce[i]/dic_dom_all[i] <1 or dic_dom_all[i]>1:\n",
    "            l.append(i)\n",
    "            BDCD[i]=dic_dom_bounce[i]/dic_dom_all[i]\n",
    "            \n",
    "        \n",
    "    return(BDCD,l,dic_dom_bounce)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Hacer una funcion para identificar los 10 dominios con un tassa de bounce abajo de 0.95 que\n",
    "#generan lo mas bounces."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "66.96131\n"
     ]
    }
   ],
   "source": [
    "#1min\n",
    "t1=time.clock()\n",
    "BPD=bounce_por_dominios(df_base)\n",
    "t2=time.clock()\n",
    "print(t2-t1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#salir los max y hacer un dictionario con el numero de bounce y el numbre del dominio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#retorna los dominios bounced por mas de 0.95 y la lista del numero de bounces que generan\n",
    "def listas_95(dic_bounce, lista_bounce, dic_dom_bounce):\n",
    "\n",
    "    l_1=[]\n",
    "    for i in lista_bounce:\n",
    "        if dic_bounce[i]>= 0.95:\n",
    "            l_1.append(i)\n",
    "            \n",
    "    l_2=[]\n",
    "    for i in l_1:\n",
    "        #print(dic_dom_bounce[i])\n",
    "        l_2.append(dic_dom_bounce[i])\n",
    "        \n",
    "    return(l_1, l_2)\n",
    "\n",
    "def dom_mas_bounce(lista_num):\n",
    "    \n",
    "    l_index=[]\n",
    "    l_2=lista_num\n",
    "    for i in range(10):\n",
    "        m=max(l_2)\n",
    "        print(m)\n",
    "        #print(listas_95(dic_dom_bounce)[1].index(m))\n",
    "        #print(listas_95(dic_dom_bounce)[0][listas_95(dic_dom_bounce)[1].index(m)])\n",
    "        l_index.append(listas_95(BPD[0],BPD[1],BPD[2])[0][listas_95(BPD[0],BPD[1],BPD[2])[1].index(m)])\n",
    "        l_2.remove(m)\n",
    "    \n",
    "    return l_index\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "233\n",
      "183\n",
      "149\n",
      "101\n",
      "73\n",
      "70\n",
      "70\n",
      "64\n",
      "57\n",
      "55\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['gamil.com',\n",
       " 'banconal.com.pa',\n",
       " 'cwp.net.pa',\n",
       " 'segurosg.com',\n",
       " 'seguroscentralizados.com.pa',\n",
       " 'sinfo.net',\n",
       " 'sinfo.net',\n",
       " 'mail.com',\n",
       " 'hsbc.com.pa',\n",
       " 'latinmail.com']"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dom_mas_bounce(listas_95(BPD[0],BPD[1],BPD[2])[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Reduccion de las variables: ACP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Hacemos la PCA con la base vectorizada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PCA(copy=True, n_components=4, whiten=False)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "pca = PCA(n_components=4)\n",
    "print(pca.fit(df_vect))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  9.99993018e-01,   6.36712604e-06,   3.33999501e-07,\n",
       "         1.52867954e-07])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pca.explained_variance_ratio_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x10a986f98>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXkAAAEKCAYAAAD3tSVSAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAEj9JREFUeJzt3GuUXWV9x/HvxABCMglEx0utd+BfLIqWsAIxArbE1UpY\nxlvbWEGj8UK94IKyJHZZLS3ejaIVAUNRXC5aGxuttyBF8DIgVqQab/9g0/aFYpkymItcmpjpi71H\nN+PMnDMze5IzD9/PG2bvZ59nfuch+Z2dfc4+fSMjI0iSyjTvQAeQJM0eS16SCmbJS1LBLHlJKpgl\nL0kFs+QlqWCWvGYsIr4cEW8cZ/95EfHpKc711xHx4vbSHTgRcUpEbK1/buV5RcRjIuJnEfHkmSfU\nA8H8Ax1ARfgQcBHwzjH71wGvncpEmfmWtkL1iBFo9Xk9EliVmVtbmk+Fs+TVhk8D74+Ip2fmIFRn\nsQCZeV1E9AHvA5YB/UAfsC4zb4qIK4ElwBOAzwGPALZm5oaIeBnwSuCg+ph3ZOZlEfES4LnAPuAo\n4D7grMz8QUQ8HLgU+B3gl8BlmfnBiFgEXAwcW893HXB+Zu5rPpGIOIjqxepk4EHArcDrgUOB7wAv\ny8wtEXFh/Xz+ENher8EKYDGwITMvHTPvlY3n9XzgQuBu4AvAmzLzoPp5vSAzz6gf86vtsbki4lbg\n9Zm5OyJ+C/g74NH1c/uHzHzHlP4PqlhertGMZeYvgY9QnbmPegVwSf3zMuCRmXlSZh4LXAVc0Dj2\n0Mx8cmauH90REQuAlwN/lJnHA38KvLvxmJOB12Tmk4EbgfPr/R+uIuUxwHLgFRHxBKoXmW9l5gnA\n7wEDwHnjPJ0LgD2ZuTQznwbcDrwzM4eAlwCXR8RzgLOANZk5esv4wnruU4ELI+J3x1uriHgEcAXw\nvPr4vdz/7+HYW9BHt8fNVY99HLiinm8ZsDIiXjDe79cDj2fyasvlwPfrcj4EeBZwNkBmfiMi3hwR\nrwaeSFWEOxuP/frYyTLzFxFxBrAqIo4CngosaBxyS2beXv/8baoze4A/AP6inmMn8BSAiFgFnBAR\noy9ED+Y3CxVgFbA4Ip5Vbx8E/E8937UR8Ungn4FnZOZw43Efqo/5aURsqZ//t8eZ/+nAdzIz6+0P\nAG8d57iuckXEYcApwBER8bf12AKq9drUxbwqnCWvVmTmzyLiWmANVclsysxdABFxOvB+4D1UlzV+\nBPxZ4+G7x84XEY8CbgIuA75GVVinNw65p/HzCNUlIKjOjH9V3hHxOOBOqrPlF46Wa0QsZvySfxBw\nTmZeUx+3gOoFYdSTgJ9R/Svhxsb+vY2f51FdKhrPPdz/zH3PBM8D4OAucj2oHj8pM++rxx5KdSlI\n8nKNWvVhqvI+i/rMtnYa8C+ZeRnwLWA1vy6niSwF7sjMizLzWmD0OnXf5A/jWmBtfexi4MvAkcA1\nwLkR0RcRBwObgdeM8/hrgNdGxMERMY/qReZt9XznAofV2c6NiOMbjzurPuYxwErgixPkuwk4MiKe\nWm+vbYwNAcfWv3v+6HOeLFf9QvoN6n+91M/5q8BzJlkjPYBY8mpNZn4FeAiwIzO/3xi6FDg1Ir5N\n9UbjtcDjJ5hm9Oz6GuAnEZER8VXgXqoz6CM7xHgd8KSI+A7VvwAuysxbgXOoCvq7VG+gfg941ziP\n/xvgv6jecP0e1d+R8+pSvoDqDd7bgTcAV0fEwvpxj4mIW6jK/ZzMvG28cJl5F/DHwMaI+DfghMbw\nl4CvAFn/97udctVjLwJOjIjvUhX+1Zl59eTLpAeKPr9qWJqZiPhP4E8y85vTeOzDgdsz0xMuzYqu\n/mBFxLKIuH6c/WdExDcjYrDxhpb0QDPTMyXPtDRrOp7JR8T5wJnA7sxc3tg/H/ghcDzVm0mDwOn1\nR80kST2gmzP5H/Prj6c1HQPclpk7M3MP1cfgTm4znCRpZjqWfGZu5v4fDxu1CNjR2N5FdbefJKlH\nzORz8jupin5UP/DzTg9a9vy3jhy2+GEz+LVluHvHHXz87S/i6KOPPtBRJM0NnT4+PK6plPzYX/BD\nqs/7Hk5148XJ3P+283EdtvhhLDziUVP4teUaHt7N0NCuGc0xMNA/4zn2B3O2Zy5kBHO2bWCgf1qP\nm0rJjwBExBpgQWZurG8O+RLVC8DGxm3mkqQe0FXJZ+Z/U93GTfMmi8z8PPD52YkmSZopb8CQpIJZ\n8pJUMEtekgpmyUtSwSx5SSqYJS9JBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUv\nSQWz5CWpYJa8JBXMkpekglnyklQwS16SCmbJS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJU\nMEtekgpmyUtSwSx5SSqYJS9JBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsHmdzogIvqAS4Dj\ngHuBdZm5vTH+XOBNwD7gysy8dJaySpKmqJsz+dXAIZm5HFgPbBgzvgE4DVgBnBcRi9uNKEmarm5K\nfgWwBSAzbwaWjhn/P+AI4NB6e6S1dJKkGemm5BcBOxrbeyOi+bj3ArcAW4HPZebOFvNJkmag4zV5\nYCfQ39iel5n7ACLi0cDrgMcCvwA+ERHPz8xPtZ60QEuWLGRgoL/zgR20Mcf+YM72zIWMYM5e0E3J\nDwKrgE0RcSLVGfuoBwN7gfsycyQi7qC6dKMuDA/vZmho14zmGBjon/Ec+4M52zMXMoI52zbdF6Ju\nSn4zsDIiBuvttRGxBliQmRsj4irgxoi4B/gP4KPTSiJJal3Hks/MEeDsMbu3NcbfB7yv5VySpBZ4\nM5QkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXMkpekglnyklQwS16SCmbJ\nS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5SSqYJS9JBbPkJalglrwk\nFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXMkpekglnyklQwS16SCmbJS1LB\n5nc6ICL6gEuA44B7gXWZub0xfgLw3nrzJ8BZmblnFrJKkqaomzP51cAhmbkcWA9sGDN+OfDSzDwZ\nuA54fLsRJUnT1U3JrwC2AGTmzcDS0YGIOBq4Ezg3Im4ADs/MbbOQU5I0Dd2U/CJgR2N7b0SMPu6h\nwEnAB4DTgNMi4tRWE0qSpq3jNXlgJ9Df2J6Xmfvqn+8Efjx69h4RW6jO9G9oM2SplixZyMBAf+cD\nO2hjjv3BnO2ZCxnBnL2gm5IfBFYBmyLiRGBrY2w7sDAinlC/GfsMYGP7Mcs0PLyboaFdM5pjYKB/\nxnPsD+Zsz1zICOZs23RfiLop+c3AyogYrLfXRsQaYEFmboyIlwNXRwTAjZn5xWklkSS1rmPJZ+YI\ncPaY3dsa4zcAy9qNJUlqgzdDSVLBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtS\nwSx5SSqYJS9JBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXM\nkpekglnyklQwS16SCmbJS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5\nSSqYJS9JBbPkJalglrwkFWx+pwMiog+4BDgOuBdYl5nbxznuMuDOzHxT6yklSdPSzZn8auCQzFwO\nrAc2jD0gIl4FHNtyNknSDHVT8iuALQCZeTOwtDkYEScBJwCXtZ5OkjQj3ZT8ImBHY3tvRMwDiIhH\nAG8BXgv0tR9PkjQTHa/JAzuB/sb2vMzcV//8QuAhwBeARwKHRsSPMvOqdmOWacmShQwM9Hc+sIM2\n5tgfzNmeuZARzNkLuin5QWAVsCkiTgS2jg5k5geBDwJExEuAsOC7Nzy8m6GhXTOaY2Cgf8Zz7A/m\nbM9cyAjmbNt0X4i6KfnNwMqIGKy310bEGmBBZm6c1m+VJO0XHUs+M0eAs8fs3jbOcR9rK5QkqR3e\nDCVJBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXMkpekglny\nklQwS16SCmbJS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5SSqYJS9J\nBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXMkpekgs3vdEBE\n9AGXAMcB9wLrMnN7Y3wNcA6wB9iamX8+S1klSVPUzZn8auCQzFwOrAc2jA5ExIOBC4FTMvMZwOER\nsWpWkkqSpqybkl8BbAHIzJuBpY2x+4DlmXlfvT2f6mxfktQDuin5RcCOxvbeiJgHkJkjmTkEEBGv\nAxZk5r+2H1OSNB0dr8kDO4H+xva8zNw3ulFfs38XcBTwvHbjlW3JkoUMDPR3PrCDNubYH8zZnrmQ\nEczZC7op+UFgFbApIk4Eto4Zvxy4JzNXtx2udMPDuxka2jWjOQYG+mc8x/5gzvbMhYxgzrZN94Wo\nm5LfDKyMiMF6e239iZoFwC3AWuBrEXE9MAJcnJmfmVYaSVKrOpZ8Zo4AZ4/ZvW0qc0iSDgxvhpKk\nglnyklQwS16SCmbJS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5SSqY\nJS9JBbPkJalglrwkFcySl6SCWfKSVDBLXpIKZslLUsEseUkqmCUvSQWz5CWpYJa8JBXMkpekglny\nklQwS16SCmbJS1LBLHlJKpglL0kFs+QlqWCWvCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5SSrY/E4H\nREQfcAlwHHAvsC4ztzfGzwDeDOwBrszMjbOUVZI0Rd2cya8GDsnM5cB6YMPoQETMr7dPA04FXhkR\nA7OQU5I0Dd2U/ApgC0Bm3gwsbYwdA9yWmTszcw/wdeDk1lNKkqal4+UaYBGwo7G9NyLmZea+ccZ2\nAYsnm+zuHXdMOWSJ2lqHbdu2MTy8u5W5ZtNddy2c9ZxPfOJRM55jLqzn/lhLmPl6zoW1hLnzZ3O6\nuin5nUB/Y3u04EfHFjXG+oGfTzbZzZ96a9+UEmpSAwP9nQ9S11zP9riWvaGbyzWDwLMBIuJEYGtj\n7IfAkRFxeEQcTHWp5qbWU0qSpqVvZGRk0gMan655Sr1rLXA8sCAzN0bE6cBbgD7gisy8dBbzSpKm\noGPJS5LmLm+GkqSCWfKSVDBLXpIK1s1HKKeli69DeAOwDhj9wPirMvO22cozmYhYBrwjM585Zn9P\nfWXDJDl7Yi3rO6D/HngccDBwUWZ+tjHeE+vZRc5eWc95wEeAAPYBr87MHzTGe2U9O+XsifWsszwM\n+BZwWmZua+zvibVs5Jko55TXctZKnsbXIdTltKHeN+p44MzMvHUWM3QUEecDZwK7x+wf/cqG44F7\ngMGI+ExmDu3/lBPnrPXEWgIvBv43M8+KiCOAfwc+Cz23nhPmrPXKep4BjGTmiog4BXgb9d+hHlvP\nCXPWemI96zW7FLh7nP29spYT5qxNeS1n83LNZF+HAFXY9RHxtYi4YBZzdPJj4Lnj7O+1r2yYKCf0\nzlp+kupsCKo/W3saY720npPlhB5Zz8z8DPDKevNxwF2N4Z5Zzw45oUfWE3gP8GHgp2P298xa1ibK\nCdNYy9ks+XG/DqGxfTXwauCZwIqIePYsZplQZm4G9o4zNOWvbJhNk+SE3lnLuzPzFxHRD/wT8JeN\n4Z5Zzw45oUfWEyAz90XElcDFwCcaQz2znjBpTuiB9YyIlwJ3ZOa1VPf0NPXMWnbICdNYy9ks+cm+\nDgHg4swczsy9wOeBp81ilumY8lc2HEA9s5YR8Wjgy8DHMvMfG0M9tZ6T5IQeWk+AzFwLHA1sjIhD\n6909tZ4wYU7ojfVcC6yMiOuBpwJX1de9obfWcrKcMI21nM1r8oPAKmDT2K9DiIhFwNaIOIbqGtjv\nA1fMYpZujH3V/NVXNlBdGzsZePd+T/Wb7pezl9YyIh4OXAO8JjOvHzPcM+s5Wc4eW88zgd/OzLdT\nfXjhl1RvbEJvreeEOXtlPTPzlEbe66nesBx987Jn1nKynNNdy9ks+c1Ur0iD9fbaiFjDr78O4Y3A\nDVR/KK7LzC2zmKUbIwBjMp4LfImqWDdm5u0HMmBtvJy9spbrgcOBN0fEX9VZP0LvrWennL2ynpuA\nj0bEV6j+rr4BeF5E9Np6dsrZK+s56gH1d92vNZCkgnkzlCQVzJKXpIJZ8pJUMEtekgpmyUtSwSx5\nSSqYJS9JBbPkJalg/w9Ek1IwVtsqyQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1026cb240>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.bar(numpy.arange(len(pca.explained_variance_ratio_))+0.5, pca.explained_variance_ratio_)\n",
    "plt.title(\"Variance expliquée\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Necesitamos de normalizar la base para que la PCA sea mas justa. Pero no se si deberia hacerla\n",
    "#solo con las variables explicativas o tambien con la variable que debemos explicar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x10bfcfd30>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1026cb860>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.preprocessing import normalize\n",
    "xnorm = normalize(df_vect)\n",
    "pca = PCA(n_components=6)\n",
    "pca.fit(xnorm)\n",
    "plt.bar(numpy.arange(len(pca.explained_variance_ratio_))+0.5, pca.explained_variance_ratio_)\n",
    "plt.title(\"Variance expliquée\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "76.88032600000001\n"
     ]
    }
   ],
   "source": [
    "#1min17s\n",
    "t4=time.clock()\n",
    "print(t4-t3)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.4.2"
  },
  "widgets": {
   "state": {},
   "version": "1.1.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
