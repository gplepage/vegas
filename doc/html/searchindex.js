Search.setIndex({"docnames": ["background", "c_fortran", "index", "outliers", "tutorial", "vegas"], "filenames": ["background.rst", "c_fortran.rst", "index.rst", "outliers.rst", "tutorial.rst", "vegas.rst"], "titles": ["How <code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">vegas</span></code> Works", "Integrands in C or Fortran", "vegas Documentation", "Case Study: Bayesian Curve Fitting", "Tutorial", "<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">vegas</span></code> Module"], "terms": {"integr": [0, 1, 2, 3], "adaptivemap": [0, 1, 2, 4], "runningwavg": [0, 1, 4], "chi": [0, 1, 3, 4, 5], "2": [0, 1, 3, 4, 5], "x": [0, 1, 3, 4, 5], "y": [0, 1, 3, 4, 5], "s": [0, 1, 3, 4, 5], "1": [0, 1, 3, 4, 5], "m": [0, 1, 5], "sigma_i": [0, 3], "m_": [0, 4], "mathrm": [0, 4], "st": [0, 4], "d_0": [0, 4], "d": [0, 1, 4, 5], "us": [0, 1, 3, 4, 5], "two": [0, 3, 4, 5], "strategi": [0, 3, 4], "here": [0, 1, 3, 4, 5], "we": [0, 1, 3, 4, 5], "discuss": [0, 1, 3, 4], "idea": [0, 4, 5], "behind": 0, "each": [0, 3, 4, 5], "turn": [0, 4, 5], "most": [0, 3, 4, 5], "its": [0, 3, 4, 5], "remap": [0, 4, 5], "variabl": [0, 3, 4, 5], "direct": [0, 4, 5], "befor": [0, 3, 4, 5], "make": [0, 1, 3, 4, 5], "mont": [0, 3, 4, 5], "carlo": [0, 3, 4, 5], "estim": [0, 1, 3, 4, 5], "thi": [0, 1, 3, 4, 5], "equival": [0, 4, 5], "standard": [0, 3, 4, 5], "optim": [0, 1, 3, 4, 5], "call": [0, 1, 3, 4, 5], "choos": [0, 4, 5], "transform": [0, 4, 5], "minim": [0, 4, 5], "statist": [0, 4, 5], "error": [0, 3, 4, 5], "whose": [0, 4, 5], "integrand": [0, 2, 3, 5], "ar": [0, 1, 3, 4, 5], "uniformli": [0, 4, 5], "distribut": [0, 1, 2, 3, 5], "new": [0, 1, 3, 4, 5], "one": [0, 3, 4, 5], "dimens": [0, 4, 5], "exampl": [0, 3, 4, 5], "replac": [0, 1, 3, 4, 5], "origin": [0, 1, 4, 5], "over": [0, 3, 4, 5], "i": [0, 1, 3, 4, 5], "int_a": 0, "b": [0, 3, 4, 5], "dx": [0, 4, 5], "f": [0, 1, 3, 4, 5], "an": [0, 1, 3, 4, 5], "int_0": [0, 4], "dy": [0, 5], "j": [0, 1, 4, 5], "where": [0, 1, 3, 4, 5], "jacobian": [0, 2, 5], "A": [0, 1, 2, 4, 5], "simpl": [0, 4, 5], "given": [0, 4, 5], "approx": [0, 4, 5], "equiv": [0, 4], "frac": [0, 4], "sum_i": [0, 4], "sum": [0, 2], "random": [0, 1, 2, 3, 5], "point": [0, 1, 3, 4, 5], "between": [0, 4, 5], "0": [0, 1, 3, 4, 5], "itself": [0, 4, 5], "number": [0, 2, 5], "from": [0, 1, 3, 4, 5], "mean": [0, 3, 4, 5], "exact": [0, 4], "varianc": [0, 4, 5], "left": [0, 5], "right": [0, 3, 5], "deviat": [0, 3, 4, 5], "possibl": [0, 3, 4, 5], "straightforward": [0, 1, 4], "variat": [0, 2], "calcul": [0, 2, 3, 5], "constrain": 0, "show": [0, 1, 3, 4, 5], "Such": [0, 1, 4, 5], "greatli": [0, 4], "reduc": [0, 3, 4, 5], "when": [0, 3, 4, 5], "ha": [0, 1, 3, 4, 5], "high": [0, 3, 4, 5], "peak": [0, 4, 5], "sinc": [0, 1, 4, 5], "propto": 0, "region": [0, 4, 5], "space": [0, 4, 5], "larg": [0, 4, 5], "stretch": [0, 5], "out": [0, 1, 3, 4, 5], "consequ": 0, "uniform": [0, 3, 4, 5], "place": [0, 3, 4, 5], "more": [0, 1, 3, 4, 5], "than": [0, 1, 3, 4, 5], "would": [0, 3, 4, 5], "were": [0, 3, 5], "concentr": [0, 4, 5], "which": [0, 1, 3, 4, 5], "why": [0, 3], "product": [0, 3, 4], "becom": [0, 4], "gaussian": [0, 3, 4, 5], "limit": [0, 4, 5], "non": [0, 3, 4, 5], "correct": [0, 4], "vanish": [0, 4], "like": [0, 1, 4, 5], "For": [0, 3, 4, 5], "easi": [0, 1, 3, 4], "langl": [0, 4], "4": [0, 1, 3, 4, 5], "rangl": [0, 4], "3": [0, 1, 3, 4, 5], "moment": [0, 4], "equal": [0, 4, 5], "fall": [0, 4, 5], "wa": [0, 4], "result": [0, 1, 2, 3, 5], "so": [0, 1, 4, 5], "neglig": [0, 4], "These": [0, 3, 4, 5], "assum": [0, 5], "n": [0, 1, 3, 4, 5], "all": [0, 3, 4, 5], "need": [0, 1, 4, 5], "case": [0, 2, 4, 5], "singular": [0, 4], "implement": [0, 1, 2, 3], "grid": [0, 4, 5], "x_0": [0, 4, 5], "x_1": [0, 4, 5], "delta": [0, 4, 5], "x_2": [0, 5], "cdot": [0, 5], "x_n": [0, 5], "x_": [0, 5], "specifi": [0, 1, 3, 4, 5], "function": [0, 1, 2, 3, 4], "ldot": [0, 4, 5], "x_i": [0, 5], "linear": [0, 3, 5], "interpol": [0, 5], "those": [0, 1, 4, 5], "piecewis": 0, "constant": [0, 4, 5], "j_i": [0, 5], "int_": [0, 4], "treat": 0, "independ": [0, 4, 5], "constraint": 0, "y_i": 0, "trivial": 0, "mbox": 0, "adjust": [0, 4, 5], "until": [0, 4, 5], "last": [0, 3, 4, 5], "condit": [0, 5], "satisfi": 0, "As": [0, 3, 4, 5], "increment": [0, 5], "small": [0, 4, 5], "typic": [0, 1, 4, 5], "knowledg": 0, "initi": [0, 4, 5], "start": [0, 4, 5], "also": [0, 1, 3, 4, 5], "inform": [0, 4, 5], "refin": [0, 4], "choic": 0, "bring": 0, "them": [0, 3, 4, 5], "closer": [0, 3], "valu": [0, 1, 3, 4, 5], "subsequ": [0, 4, 5], "iter": [0, 1, 3, 4, 5], "usual": [0, 1, 3, 4, 5], "converg": [0, 4, 5], "after": [0, 4, 5], "sever": [0, 4, 5], "analysi": [0, 3, 4, 5], "gener": [0, 1, 2, 3, 5], "easili": [0, 3, 4], "multi": [0, 4, 5], "dimension": [0, 4, 5], "appli": [0, 4, 5], "similar": [0, 1, 3, 4], "along": [0, 4, 5], "axi": [0, 3, 4, 5], "made": [0, 1, 3, 4, 5], "smaller": [0, 4, 5], "project": [0, 4, 5], "onto": [0, 3, 4, 5], "larger": [0, 3, 4, 5], "four": 0, "section": [0, 1, 4], "basic": [0, 2], "look": [0, 3], "same": [0, 1, 3, 4, 5], "evalu": [0, 1, 3, 4, 5], "averag": [0, 1, 3, 4, 5], "everi": [0, 4, 5], "rectangl": 0, "pictur": 0, "much": [0, 3, 4, 5], "less": [0, 3, 4, 5], "higher": 0, "therefor": [0, 4], "numer": 0, "vicin": 0, "5": [0, 1, 3, 4, 5], "plot": [0, 5], "obtain": [0, 3, 4], "includ": [0, 1, 3, 4, 5], "line": [0, 3, 4], "integ": [0, 1, 4, 5], "show_grid": [0, 5], "30": [0, 3, 4, 5], "code": [0, 1, 3, 4, 5], "finish": [0, 4], "It": [0, 1, 3, 4, 5], "caus": [0, 3, 4, 5], "matplotlib": [0, 4, 5], "instal": [0, 1, 4, 5], "creat": [0, 1, 3, 4, 5], "imag": 0, "locat": [0, 4, 5], "node": [0, 5], "99": [0, 4], "too": [0, 3, 4, 5], "mani": [0, 1, 3, 4], "displai": [0, 5], "low": [0, 4], "resolut": 0, "obviou": [0, 3], "follow": [0, 1, 3, 4, 5], "arrang": 0, "diagon": [0, 4], "hypercub": [0, 4, 5], "math": [0, 1, 4], "def": [0, 1, 3, 4, 5], "f2": [0, 5], "dx2": [0, 4], "rang": [0, 1, 4, 5], "exp": [0, 1, 3, 4, 5], "100": [0, 1, 4, 5], "1013": [0, 4], "2167575422921535": 0, "return": [0, 1, 3, 4, 5], "nitn": [0, 1, 3, 4, 5], "10": [0, 1, 3, 4, 5], "neval": [0, 1, 3, 4, 5], "4e4": 0, "print": [0, 1, 3, 4, 5], "q": [0, 1, 3, 4, 5], "2f": [0, 4], "70": [0, 1, 3, 4], "give": [0, 1, 3, 4, 5], "now": [0, 3, 4], "around": [0, 4], "33": [0, 1, 3, 4], "67": [0, 1, 3, 4], "unfortun": 0, "veri": [0, 3, 4, 5], "close": [0, 4, 5], "zero": [0, 3, 4, 5], "There": [0, 1, 3, 4, 5], "14": [0, 1, 3, 4, 5], "phantom": 0, "emphas": [0, 4, 5], "addit": [0, 1, 4], "actual": [0, 3, 4], "better": [0, 3, 4], "obvious": [0, 1], "wast": 0, "resourc": 0, "occur": [0, 3, 4], "becaus": [0, 3, 4, 5], "separ": [0, 1, 3, 4], "appear": 0, "have": [0, 1, 3, 4, 5], "focu": 0, "both": [0, 3, 4, 5], "what": [0, 3, 4], "doe": [0, 3, 4, 5], "orient": 0, "other": [0, 1, 2, 3, 4], "altern": [0, 4, 5], "complic": [0, 1, 4], "expens": [0, 4, 5], "princip": 0, "proven": 0, "effect": [0, 3, 4], "realist": [0, 4], "applic": 0, "alwai": [0, 4, 5], "difficulti": [0, 4], "structur": [0, 4], "lie": 0, "volum": [0, 4, 5], "To": [0, 1, 4, 5], "address": [0, 1, 4], "problem": [0, 2, 4, 5], "version": [0, 1, 4, 5], "introduc": [0, 4], "second": [0, 1, 3, 4, 5], "base": [0, 4, 5], "upon": [0, 4, 5], "anoth": [0, 1, 4, 5], "techniqu": 0, "divid": [0, 4, 5], "stratif": [0, 2, 5], "do": [0, 1, 4, 5], "ad": [0, 1, 3, 4], "togeth": [0, 4, 5], "provid": [0, 1, 4, 5], "entir": [0, 4, 5], "coarser": 0, "least": [0, 3, 4, 5], "must": [0, 1, 4, 5], "keep": [0, 4], "can": [0, 1, 3, 4, 5], "restrict": [0, 4], "older": [0, 1], "howev": [0, 3, 4], "In": [0, 3, 4, 5], "proport": [0, 3, 4, 5], "about": [0, 1, 3, 4, 5], "set": [0, 4, 5], "next": [0, 4], "wai": [0, 1, 3, 4, 5], "potenti": [0, 4, 5], "largest": [0, 4, 5], "abov": [0, 1, 3, 4], "shift": 0, "awai": [0, 3, 5], "occupi": 0, "real": [0, 1, 4, 5], "come": [0, 4], "improv": [0, 4, 5], "abil": [0, 4, 5], "contribut": [0, 4], "enough": [0, 4, 5], "permit": 0, "With": [0, 4], "factor": [0, 3, 4], "rel": [0, 1, 3, 4, 5], "differ": [0, 3, 4, 5], "difficult": [0, 3, 4, 5], "vega": [1, 3], "algorithm": [1, 3, 4, 5], "been": [1, 4, 5], "extens": 1, "The": [1, 2, 4, 5], "python": [1, 3, 4, 5], "describ": [1, 3, 4, 5], "power": 1, "combin": [1, 3, 4, 5], "substanti": [1, 4, 5], "faster": [1, 2, 3, 5], "directli": [1, 4], "thei": [1, 3, 4, 5], "speed": [1, 4, 5], "access": [1, 4], "review": 1, "few": [1, 4], "option": [1, 4, 5], "simplest": 1, "modul": [1, 2, 3, 4], "illustr": [1, 4, 5], "consid": [1, 4, 5], "written": [1, 4], "store": [1, 4], "file": [1, 4, 5], "cfcn": 1, "h": [1, 4], "doubl": [1, 4, 5], "fcn": [1, 3, 4, 5], "int": [1, 4, 5], "dim": [1, 4, 5], "xsq": 1, "sqrt": [1, 3, 4, 5], "pow": 1, "run": [1, 3, 4, 5], "cfcn_cffi": 1, "compil": [1, 4], "builder": 1, "py": [1, 4], "import": [1, 2, 3, 4, 5], "ffi": 1, "ffibuild": 1, "etc": 1, "avail": [1, 3, 4, 5], "cdef": [1, 4], "build": 1, "set_sourc": 1, "module_nam": 1, "sourc": [1, 4], "contain": [1, 3, 4, 5], "librari": 1, "mai": [1, 5], "lm": 1, "__name__": [1, 3, 4, 5], "__main__": [1, 3, 4, 5], "verbos": 1, "true": [1, 4, 5], "wrap": 1, "lib": 1, "_x": 1, "cast": [1, 4], "ctype": 1, "data": [1, 3, 4, 5], "pointer": 1, "main": [1, 3, 4, 5], "1e6": [1, 5], "summari": [1, 3, 4, 5], "output": [1, 3, 4, 5], "adapt": [1, 2, 3, 4, 5], "final": [1, 3, 4, 5], "itn": [1, 3, 4, 5], "wgt": [1, 4, 5], "chi2": [1, 3, 4, 5], "dof": [1, 3, 4, 5], "00": [1, 3, 4, 5], "7": [1, 3, 4, 5], "403": [1, 3], "42": [1, 3, 4, 5], "401": 1, "52": [1, 4, 5], "11": [1, 3, 4, 5], "366": 1, "27": [1, 3, 4, 5], "376": 1, "23": [1, 3, 4, 5], "51": [1, 3, 4], "22": [1, 3, 4, 5], "4041": 1, "73": [1, 3, 4], "4014": 1, "48": [1, 3, 4, 5], "4046": 1, "36": [1, 3, 4], "4039": 1, "32": [1, 3, 4], "15": [1, 3, 4, 5], "6": [1, 3, 4, 5], "4003": 1, "4015": 1, "19": [1, 3, 4, 5], "09": [1, 4], "37": [1, 3, 4, 5], "4036": 1, "20": [1, 3, 4, 5], "4025": 1, "01": [1, 3, 4], "8": [1, 3, 4, 5], "4017": 1, "16": [1, 3, 4, 5], "4022": 1, "89": [1, 3, 4], "9": [1, 3, 4, 5], "4010": 1, "40174": 1, "83": [1, 3, 4, 5], "84": [1, 4, 5], "57": [1, 4], "13": [1, 3, 4], "74": [1, 3, 4], "4016": 1, "12": [1, 3, 4, 5], "4030": 1, "40239": 1, "81": [1, 3, 4, 5], "79": [1, 3, 4], "4020": 1, "40224": 1, "63": [1, 4, 5], "44": [1, 3, 4], "64": [1, 3, 4], "40249": 1, "92": [1, 3, 4], "40232": 1, "31": [1, 3, 4, 5], "82": [1, 3, 4], "40258": 1, "86": [1, 3, 4], "25": [1, 3, 4], "91": [1, 3, 4, 5], "40093": 1, "40205": 1, "39": [1, 3, 4, 5], "62": [1, 4, 5], "40228": 1, "76": [1, 4, 5], "40210": 1, "35": [1, 3, 4], "60": [1, 3, 4], "40276": 1, "72": [1, 3, 4], "40222": 1, "61": [1, 4, 5], "75": [1, 3, 4, 5], "40181": 1, "71": [1, 3, 4], "40216": 1, "29": [1, 3, 4, 5], "80": [1, 3, 4, 5], "40178": 1, "53": [1, 3, 4], "85": [1, 3, 4, 5], "000": [1, 4, 5], "time": [1, 3, 4, 5], "accur": [1, 3, 4], "first": [1, 3, 4, 5], "convert": [1, 4], "batch": [1, 3, 4, 5], "void": 1, "batch_fcn": 1, "numpi": [1, 3, 4, 5], "np": [1, 3, 4, 5], "batchintegrand": [1, 4, 5], "batch_f": [1, 4], "shape": [1, 3, 4, 5], "empti": [1, 4, 5], "float": [1, 4, 5], "_an": 1, "ident": 1, "wrapper": [1, 5], "slower": [1, 4], "ffcn": 1, "end": [1, 4, 5], "o": 1, "someth": [1, 4], "gfortran": 1, "script": [1, 3, 4], "previou": [1, 4], "onli": [1, 3, 4, 5], "three": [1, 3, 4, 5], "modif": 1, "list": [1, 3, 4, 5], "object": [1, 2, 4], "extra_object": 1, "name": [1, 4], "extra": [1, 4, 5], "underscor": 1, "depend": [1, 4], "unix": [1, 4, 5], "system": 1, "nm": 1, "argument": [1, 4, 5], "pass": [1, 5], "modifi": [1, 3, 4, 5], "ffcn_cffi": 1, "fcn_": 1, "exactli": [1, 4, 5], "see": [1, 4, 5], "seed": 1, "fast": [1, 4, 5], "correspond": [1, 3, 4, 5], "flexibl": [1, 4], "interfac": 1, "increas": [1, 3, 4, 5], "effici": [1, 4], "pyx": [1, 4], "again": [1, 4], "extern": 1, "type": [1, 4, 5], "tell": [1, 4, 5], "how": [1, 2, 4, 5], "construct": [1, 4, 5], "done": [1, 3, 4, 5], "pyxbld": 1, "make_ext": 1, "modnam": 1, "pyxfilenam": 1, "distutil": 1, "include_dir": 1, "get_includ": 1, "make_setup_arg": 1, "dict": [1, 3, 4, 5], "pyximport": [1, 4], "inplac": [1, 4], "1e4": [1, 4], "guarante": [1, 4], "work": [1, 2, 4, 5], "analog": [1, 5], "packag": [1, 5], "you": [1, 4, 5], "somewhat": [1, 3, 4], "insid": 1, "part": [1, 4], "form": [1, 5], "subroutin": 1, "nbatch": 1, "xi": 1, "cf2py": 1, "intent": 1, "take": [1, 3, 4, 5], "input": 1, "arrai": [1, 3, 4, 5], "comment": 1, "should": [1, 4, 5], "correpond": 1, "automat": [1, 2, 3, 5], "deduc": 1, "ffcn_f2py": 1, "roughli": 1, "twice": [1, 3, 4], "content": 2, "tutori": 2, "introduct": 2, "multipl": [2, 5], "simultan": 2, "bayesian": [2, 5], "processor": [2, 5], "save": [2, 5], "map": [2, 3, 5], "precondit": 2, "note": [2, 3, 5], "c": [2, 3, 4, 5], "fortran": [2, 4, 5], "cffi": 2, "cython": [2, 4, 5], "f2py": 2, "sampl": [2, 4, 5], "stratifi": [2, 4, 5], "studi": [2, 4], "curv": [2, 4], "fit": [2, 4], "solut": [2, 4], "pdfintegr": [2, 3, 4], "index": [2, 3, 4, 5], "search": 2, "page": 2, "gvar": [3, 4, 5], "lsqfit": 3, "nonlinear_fit": 3, "bufferdict": [3, 4, 5], "straight": [3, 5], "outlier": 3, "special": [3, 4, 5], "probabl": [3, 4, 5], "densiti": [3, 4, 5], "pdf": [3, 4, 5], "paramet": [3, 4, 5], "jake": 3, "vanderpla": 3, "hi": 3, "blog": 3, "document": [3, 4, 5], "want": [3, 4, 5], "extrapol": 3, "figur": 3, "squar": [3, 4, 5], "dot": [3, 4], "unconvinc": 3, "particular": [3, 4], "while": [3, 4, 5], "suggest": 3, "intercept": 3, "cours": 3, "hoc": 3, "prescript": 3, "handl": [3, 4, 5], "best": [3, 4], "situat": [3, 4], "explan": 3, "model": 3, "accordingli": 3, "might": [3, 4, 5], "know": [3, 4], "some": [3, 4, 5], "fraction": [3, 4, 5], "w": [3, 4], "our": [3, 4], "devic": 3, "malfunct": 3, "measur": [3, 4], "One": [3, 4], "c_0": 3, "c_1": 3, "weight": [3, 4, 5], "assign": [3, 4, 5], "term": [3, 4], "respect": [3, 5], "gv": [3, 4, 5], "38": [3, 4], "59": [3, 4], "88": [3, 4], "68": [3, 4], "prior": [3, 5], "make_prior": 3, "mod_pdf": 3, "modifiedpdf": 3, "fitfcn": 3, "expval": [3, 5], "10_000": 3, "warmup": [3, 4], "expect": [3, 4, 5], "g": [3, 4, 5], "p": [3, 4, 5], "rbatchintegrand": [3, 4, 5], "c_outer": 3, "none": [3, 4, 5], "c_mean": 3, "fals": [3, 4, 5], "cmean": 3, "cov": 3, "outer": 3, "t": [3, 4], "wmean": 3, "w2mean": 3, "wsdev": 3, "bmean": 3, "b2mean": 3, "bsdev": 3, "str": [3, 5], "sdev": [3, 4, 5], "bay": 3, "logbf": 3, "log": 3, "pdfnorm": [3, 4, 5], "ncombin": [3, 4], "covari": [3, 4, 5], "insignific": 3, "compar": [3, 4], "corr": 3, "evalcorr": [3, 4], "w_shape": 3, "gw": 3, "gb": 3, "class": [3, 4, 5], "account": [3, 4], "failur": 3, "__init__": [3, 4], "self": [3, 4, 5], "add": [3, 4, 5], "rbatch": 3, "els": [3, 4], "__call__": [3, 4, 5], "y_fx": 3, "data_pdf1": 3, "gaussian_pdf": 3, "data_pdf2": 3, "broaden": 3, "prior_pdf": 3, "prod": [3, 4, 5], "deriv": [3, 4, 5], "staticmethod": 3, "xmean": 3, "xvar": 3, "var": 3, "pi": [3, 4], "coeffici": 3, "breadth": 3, "consist": [3, 4], "nomin": 3, "weigth": 3, "broad": 3, "invers": [3, 4, 5], "dictionari": [3, 4, 5], "instruct": 3, "ani": [3, 4, 5], "even": [3, 4, 5], "though": [3, 4], "arbitrari": [3, 4, 5], "allow": [3, 4, 5], "get": [3, 4], "reliabl": [3, 4], "entri": 3, "correl": [3, 4, 5], "matric": [3, 4, 5], "e": [3, 4, 5], "significantli": [3, 4], "requir": [3, 4, 5], "575": 3, "41": [3, 4], "577": 3, "576": [3, 4], "513": 3, "43": [3, 4], "555": 3, "24": [3, 4, 5], "97": [3, 4, 5], "50": [3, 4, 5], "500": 3, "541": 3, "21": [3, 4, 5], "523": 3, "538": 3, "96": [3, 4], "54": [3, 4, 5], "472": 3, "527": 3, "17": [3, 4], "77": [3, 4], "520": 3, "526": 3, "498": 3, "522": 3, "87": [3, 4, 5], "457": 3, "515": 3, "489": 3, "512": 3, "28791": 3, "61962": 3, "26": [3, 4, 5], "01970": 3, "00751": 3, "003519": 3, "47": [3, 4, 5], "27044": 3, "12028": 3, "617": 3, "6607": 3, "8216": 3, "620": 3, "90174839": 3, "tabl": [3, 5], "normal": [3, 4, 5], "logarithm": 3, "evid": 3, "117": 3, "prefer": 3, "reason": [3, 4], "plausibl": 3, "dash": 3, "red": 3, "matrix": [3, 4, 5], "slope": 3, "anti": 3, "guess": 3, "rate": 3, "fail": 3, "quarter": 3, "defin": [3, 4, 5], "unchang": [3, 5], "suit": [3, 4, 5], "rather": [3, 4, 5], "still": [3, 4, 5], "2020": 3, "laptop": [3, 4], "quit": [3, 4, 5], "49": [3, 4], "98": [3, 4], "45": [3, 4, 5], "94": [3, 4], "28": [3, 4, 5], "40": [3, 4, 5], "78": [3, 4], "55": [3, 4], "58": [3, 4], "95": [3, 4], "90": [3, 4], "2915": 3, "6122": 3, "0265": 3, "0101": 3, "00484": 3, "46": [3, 4], "391": 3, "6670": 3, "394": 3, "412": [3, 5], "6614": 3, "468": 3, "497": 3, "3667": 3, "429": 3, "363": 3, "381": 3, "3891": 3, "365": 3, "3864": 3, "480": 3, "6744": 3, "386": 3, "2619": 3, "2346": 3, "2701": 3, "2636": 3, "2284": 3, "2803": 3, "2891": 3, "2503": 3, "2808": 3, "2680": 3, "2566": 3, "2601": 3, "2718": 3, "2579": 3, "2658": 3, "2614": 3, "2844": 3, "56": [3, 4], "2238": 3, "2637": 3, "104": 3, "425": 3, "612": 3, "88889515": 3, "66": [3, 4], "slightli": [3, 4, 5], "lower": [3, 4, 5], "determin": [3, 4, 5], "15x": 3, "consider": 3, "precis": [3, 4, 5], "current": [3, 5], "adequ": 3, "Not": 3, "surprisingli": [3, 4], "unambigu": 3, "almost": [3, 4], "pretti": 3, "tempt": 3, "simpli": 3, "drop": [3, 4], "clearli": [3, 4], "understand": [3, 5], "quantifi": [3, 4], "sai": [3, 4], "rest": [3, 4], "poor": [3, 4], "per": [3, 4, 5], "degre": [3, 4, 5], "freedom": [3, 4, 5], "bit": 3, "hand": [3, 4], "multidimension": [4, 5], "lepag": 4, "comput": [4, 5], "phy": 4, "1978": 4, "192": 4, "439": 4, "2021": 4, "110386": 4, "compon": [4, 5], "attempt": 4, "flatten": [4, 5], "Then": [4, 5], "easier": 4, "collect": [4, 5], "dure": 4, "assumpt": 4, "needn": 4, "analyt": 4, "nor": 4, "continu": 4, "unusu": 4, "robust": 4, "well": [4, 5], "essenti": 4, "especi": [4, 5], "lot": 4, "corner": 4, "lose": 4, "featur": [4, 5], "method": [4, 5], "accuraci": [4, 5], "suffici": 4, "practic": 4, "order": [4, 5], "verifi": 4, "inde": 4, "decad": 4, "program": 4, "languag": 4, "arxiv_2009": 4, "05112": 4, "particularli": [4, 5], "below": [4, 5], "shown": 4, "complet": [4, 5], "copi": [4, 5], "worthwhil": 4, "plai": 4, "thing": 4, "chang": [4, 5], "outermost": 4, "parenthesi": 4, "statement": 4, "__future__": 4, "print_funct": 4, "your": 4, "dx_0": 4, "dx_1": 4, "dx_2": 4, "dx_3": 4, "sum_": 4, "x_d": 4, "chosen": [4, 5], "2118364296088": 4, "1000": [4, 5], "oper": [4, 5], "produc": [4, 5], "909": 4, "968": 4, "039": 4, "69": 4, "012": 4, "929": 4, "34": [4, 5], "952": 4, "003": 4, "980": 4, "994": 4, "18": [4, 5], "988": 4, "998": 4, "9922": 4, "020": 4, "0035": 4, "011": 4, "0057": 4, "just": [4, 5], "up": 4, "140": 4, "job": 4, "fifth": 4, "eventu": 4, "stop": [4, 5], "decreas": 4, "found": [4, 5], "fulli": [4, 5], "0064": 4, "therebi": [4, 5], "earli": [4, 5], "individu": 4, "drawn": 4, "quot": 4, "overlin": 4, "i_i": 4, "sigma_": 4, "pm": 4, "If": [4, 5], "plu": 4, "minu": 4, "root": 4, "unreli": [4, 5], "criterion": 4, "could": [4, 5], "fluctuat": [4, 5], "05": 4, "indic": [4, 5], "agre": 4, "within": [4, 5], "behavior": [4, 5], "trust": 4, "ravg": [4, 5], "attribut": [4, 5], "itn_result": [4, 5], "sum_nev": [4, 5], "total": [4, 5], "avg_nev": 4, "unlik": 4, "thu": 4, "maximum": [4, 5], "cost": [4, 5], "vari": [4, 5], "860": 4, "960": 4, "either": [4, 5], "far": [4, 5], "0003": 4, "99981": 4, "beyond": 4, "check": 4, "signific": [4, 5], "due": 4, "artifact": 4, "systemat": 4, "residu": 4, "bia": 4, "insuffici": 4, "bias": 4, "quickli": [4, 5], "wrong": 4, "answer": 4, "longer": [4, 5], "mask": 4, "unaffect": 4, "0008": [4, 5], "unless": [4, 5], "1500": 4, "aren": 4, "crude": 4, "peaki": [4, 5], "often": [4, 5], "highli": 4, "length": 4, "side": [4, 5], "redefin": 4, "0011": 4, "074": 4, "250": 4, "0012": 4, "65": 4, "93": 4, "0013": 4, "874": 4, "0015": 4, "949": 4, "0021": 4, "162": 4, "08": 4, "0033": 4, "301": 4, "985": 4, "0050": 4, "484": 4, "967": 4, "0078": 4, "738": 4, "0125": 4, "1131": 4, "miss": 4, "hit": 4, "shoulder": 4, "abl": 4, "find": [4, 5], "wildli": 4, "nonsens": 4, "common": 4, "discard": [4, 5], "avoid": [4, 5], "ruin": 4, "singl": [4, 5], "step": 4, "train": [4, 5], "yield": [4, 5], "993": 4, "062": 4, "001": 4, "964": 4, "987": 4, "974": 4, "9817": 4, "990": 4, "9843": 4, "9899": 4, "999": 4, "9917": 4, "008": 4, "9953": 4, "013": [4, 5], "9977": 4, "983": 4, "9958": 4, "onc": 4, "usefulli": 4, "4933": 4, "5017": 4, "4980": 4, "04": [4, 5], "4975": 4, "4979": 4, "5059": 4, "4998": 4, "5075": 4, "5012": 4, "4907": 4, "4997": 4, "5009": 4, "5000": 4, "5082": 4, "5010": 4, "5016": 4, "4934": 4, "5006": 4, "rectangular": 4, "sphere": 4, "radiu": 4, "center": 4, "f_sph": 4, "1115": 4, "3539360527281318": 4, "992": 4, "002": [4, 5], "996": 4, "004": 4, "9973": 4, "026": 4, "0001": 4, "053": 4, "0007": 4, "035": 4, "0038": 4, "991": 4, "0014": 4, "9956": 4, "022": 4, "9966": 4, "good": [4, 5], "properli": 4, "doesn": 4, "spend": 4, "effort": 4, "challeng": 4, "instead": 4, "1e16": 4, "chanc": 4, "tini": 4, "cut": 4, "half": 4, "had": 4, "infin": 4, "relev": 4, "finit": [4, 5], "z": 4, "re": 4, "express": 4, "free": 4, "damp": [4, 5], "slow": [4, 5], "down": [4, 5], "alpha": [4, 5], "control": [4, 5], "default": [4, 5], "005": 4, "016": 4, "006": 4, "973": 4, "9967": 4, "0009": 4, "0023": 4, "0002": 4, "958": 4, "9959": 4, "notic": 4, "persist": 4, "size": [4, 5], "signal": 4, "react": 4, "encount": 4, "hold": 4, "tune": 4, "off": [4, 5], "exhibit": 4, "kind": 4, "instabl": [4, 5], "present": [4, 5], "signfic": 4, "simpler": 4, "third": 4, "unweight": [4, 5], "sort": 4, "lack": 4, "strong": 4, "stabil": [4, 5], "lead": [4, 5], "5e4": 4, "eight": 4, "sharp": [4, 5], "eq": 4, "506": [4, 5], "510": 4, "530": 4, "02": 4, "596": 4, "03": 4, "629": 4, "802": 4, "681": 4, "907": 4, "719": 4, "07": 4, "736": 4, "064": 4, "095": 4, "072": 4, "924": [4, 5], "858": 4, "995": 4, "010": 4, "092": 4, "942": 4, "077": 4, "uncertainti": [4, 5], "magnitud": 4, "887": 4, "025": 4, "reiniti": 4, "advantag": 4, "ratio": [4, 5], "cancel": 4, "sharpli": 4, "i_0": 4, "4x": 4, "200": 4, "i_1": 4, "i_2": 4, "width": [4, 5], "ax": [4, 5], "1ex": 4, "sigma_x": 4, "2000": 4, "10000": [4, 5], "ncorrel": 4, "taken": 4, "00024682": 4, "000123417": 4, "000062327": 4, "500017": 4, "0024983": 4, "98002885": 4, "92558296": 4, "98157932": 4, "8x": 4, "absent": 4, "bulk": 4, "sigma": 4, "2_x": 4, "48x": 4, "repres": [4, 5], "arithmet": 4, "http": 4, "readthedoc": 4, "io": 4, "github": 4, "com": 4, "gplepag": 4, "git": 4, "rewritten": 4, "kei": [4, 5], "descript": 4, "intellig": 4, "write": [4, 5], "maintain": 4, "di": 4, "five": [4, 5], "interv": [4, 5], "dr": 4, "tabul": 4, "rmax": 4, "appropri": [4, 5], "bin": 4, "dtype": [4, 5], "len": 4, "85040": 4, "85039": 4, "85085": 4, "85061": 4, "85105": 4, "85079": 4, "85087": 4, "85097": 4, "85091": 4, "85099": 4, "85096": 4, "85112": 4, "851013": 4, "85114": 4, "851053": 4, "85101": 4, "851041": 4, "0759": 4, "2091": 4, "3217": 4, "3209": 4, "0723": 4, "999999999996": 4, "full": 4, "roundoff": [4, 5], "reveal": 4, "820": 4, "813": 4, "837": 4, "85236": 4, "06": [4, 5], "85242": 4, "85234": 4, "85179": 4, "00012": 4, "00177": 4, "00016": 4, "00067": 4, "00103": 4, "0000117": 4, "go": 4, "clear": [4, 5], "101": 4, "suscept": 4, "break": 4, "unnecessari": [4, 5], "further": [4, 5], "85113": 4, "851130": 4, "85124": 4, "851166": 4, "85129": 4, "851197": 4, "851168": 4, "85120": 4, "851173": 4, "851169": 4, "85107": 4, "851156": 4, "851154": 4, "85110": 4, "851149": 4, "00034": 4, "00185": 4, "00101": 4, "000000000000": 4, "purpos": 4, "layout": 4, "facilit": 4, "feasibl": 4, "pre": [4, 5], "fb": [4, 5], "g_expval": 4, "fp": 4, "f_f2": 4, "fmean": 4, "fsdev": 4, "norm": 4, "uncorrel": 4, "approxim": [4, 5], "0024": 4, "9995": 4, "0000": 4, "9932": 4, "9971": 4, "0017": 4, "9981": 4, "0063": 4, "833": 4, "mostli": 4, "pure": 4, "1e3": 4, "hundr": 4, "thousand": 4, "million": 4, "mode": [4, 5], "2e5": 4, "f_batch": 4, "intern": [4, 5], "process": [4, 5], "nhcube_batch": 5, "offer": 4, "accept": 4, "label": [4, 5], "decor": [4, 5], "send": 4, "behav": 4, "built": [4, 5], "won": 4, "act": 4, "That": 4, "hybrid": 4, "cython_integrand": 4, "libc": 4, "cimport": 4, "put": 4, "rewrit": 4, "sometim": [4, 5], "conveni": 4, "rightmost": 4, "lbatchintegrand": [4, 5], "support": [4, 5], "parallel": 4, "shorten": 4, "execut": [4, 5], "costli": 4, "nproc": [4, 5], "2019": 4, "ridg": 4, "spread": [4, 5], "x0": 4, "linspac": 4, "xd": 4, "feed": 4, "overhead": [4, 5], "manag": 4, "multiprocess": [4, 5], "window": [4, 5], "maco": [4, 5], "prevent": [4, 5], "being": [4, 5], "launch": [4, 5], "spawn": 4, "detail": [4, 5], "issu": [4, 5], "linux": 4, "pickl": [4, 5], "attributeerror": 4, "interact": 4, "environ": 4, "oppos": 4, "command": 4, "platform": [4, 5], "fix": 4, "mpi": [4, 5], "via": 4, "mpi4pi": [4, 5], "omit": [4, 5], "mpirun": 4, "speedup": 4, "synchron": [4, 5], "happen": 4, "across": [4, 5], "parallelintegrand": 4, "multiprocessor": 4, "pool": 4, "super": 4, "__del__": 4, "cleanup": 4, "join": 4, "chunk": 4, "nx": [4, 5], "concaten": 4, "fparallel": 4, "cpu": 4, "core": 4, "ineffici": 4, "necessari": 4, "trick": 4, "floor": [4, 5], "evenli": [4, 5], "5x": 4, "3x": 4, "later": [4, 5], "dump": 4, "load": [4, 5], "keyword": [4, 5], "pkl": [4, 5], "00050": 4, "115": 4, "00051": 4, "00052": 4, "00053": 4, "045": 4, "00059": 4, "00069": 4, "152": 4, "023": 4, "00093": 4, "307": 4, "00141": 4, "573": 4, "00208": 4, "896": 4, "retriev": 4, "open": [4, 5], "rb": [4, 5], "ifil": [4, 5], "ones": [4, 5], "minut": 4, "hour": [4, 5], "updat": 4, "short": 4, "monitor": [4, 5], "progress": [4, 5], "through": 4, "recent": 4, "termin": 4, "crash": 4, "badli": 4, "distort": 4, "isn": 4, "sixth": 4, "redo": 4, "024": 4, "017": 4, "saveal": [4, 5], "tupl": [4, 5], "new_result": 4, "nnew": 4, "extend": [4, 5], "earlier": 4, "under": 4, "007": 4, "997": [4, 5], "0029": [4, 5], "015": 4, "0044": 4, "0136": 4, "0114": 4, "0076": 4, "0082": 4, "0047": 4, "merg": [4, 5], "old": [4, 5], "restart": 4, "rememb": 4, "readapt": 4, "known": 4, "ahead": 4, "mathbf": 4, "r": [4, 5], "_i": 4, "narrow": 4, "_1": 4, "_2": 4, "247366": 4, "171": 4, "loc": 4, "scale": [4, 5], "adapt_to_sampl": [4, 5], "alreadi": [4, 5], "984": [], "0018": 5, "9874": [], "9975": [], "contrast": 4, "without": [4, 5], "920": 4, "091": 4, "063": 4, "031": 4, "matter": 4, "cover": 4, "domin": 4, "xsampl": 4, "fx": 4, "jac": [4, 5], "invmap": [4, 5], "ysampl": 4, "fill": [4, 5], "smc": 4, "unit": [4, 5], "fy": 4, "std": 4, "50_000": 4, "3f": 4, "021": 5, "703": 4, "535": 4, "nstrat": [4, 5], "sub": 4, "By": [4, 5], "remain": [4, 5], "maxim": [4, 5], "subset": 4, "_3": 4, "align": 4, "top": 4, "ri": 4, "356047712484621": 4, "excel": 4, "014": 4, "099": 4, "0028": [4, 5], "struggl": 4, "800": 4, "637": 4, "638": 4, "755": 4, "684": 4, "750": 4, "704": 4, "716": 4, "_0": 4, "sin": 4, "le": 4, "i_a": 4, "num": 4, "den": 4, "i_b": 4, "similarli": 4, "ia_num": 4, "1e2": 4, "ia_den": 4, "ib": 4, "20000": 4, "ia": 4, "useless": 4, "160568": 4, "160551": 4, "160559": 4, "160533": 4, "160549": 4, "160555": 4, "160553": 4, "160554": 4, "0099995": 4, "0562": 4, "100x": 4, "despit": 4, "tailor": 4, "approach": 4, "dx_d": 4, "dy_d": 4, "dy_0": 4, "y_0": 4, "uses_jac": [4, 5], "160560": 4, "160547": 4, "01000103": 4, "0749": 4, "0536": 4, "97868": 4, "whatev": 4, "level": 4, "random_batch": [4, 5], "snippet": 4, "subdivid": 4, "mimic": 4, "belong": 4, "hcube": [4, 5], "yield_hcub": [4, 5], "wgt_fx": 4, "accumul": [4, 5], "idx": 4, "select": 4, "item": [4, 5], "cube": 4, "nwf": 4, "wf": 4, "sum_wf": 4, "sum_wf2": 4, "identifi": 4, "yield_i": [4, 5], "jac1d": [4, 5], "reli": 4, "graphic": 4, "pip": 4, "tricki": 4, "ravgarrai": 5, "ravgdict": 5, "central": 5, "integration_region": 5, "specif": 5, "success": 5, "peopl": 5, "care": 5, "otherwis": 5, "unstabl": 5, "analyz": 5, "report": 5, "sy": 5, "stdout": 5, "intermedi": 5, "help": 5, "long": 5, "upper": 5, "exist": 5, "posit": 5, "explicitli": 5, "neval_frac": 5, "desir": 5, "beta": 5, "redistribut": 5, "amount": 5, "theoret": 5, "ignor": 5, "bool": 5, "evalut": 5, "involv": 5, "safe": 5, "accomplish": 5, "greater": 5, "disabl": 5, "machin": 5, "os": 5, "cpu_count": 5, "begin": 5, "cummul": 5, "memori": 5, "maxinc_axi": 5, "littl": 5, "max_nhcub": [], "never": [], "1e9": 5, "ram": 5, "2014": [], "usag": 5, "minimize_mem": [4, 5], "max_neval_hcub": 5, "max_mem": 5, "memoryerror": 5, "rais": 5, "rtol": 5, "caution": 5, "atol": 5, "absolut": 5, "ran_array_gener": 5, "sync_ran": 5, "adapt_to_error": 5, "superior": 5, "uniform_nstrat": 5, "d1": 5, "d2": 5, "flag": 5, "impact": 5, "perform": 5, "workspac": 5, "grow": 5, "exce": [], "slowli": [], "storag": [], "karg": 5, "relat": 5, "reduct": 5, "loop": 5, "whole": 5, "reset": 5, "callabl": 5, "reconstruct": 5, "allresult": 5, "clone": 5, "ka": 5, "constructor": 5, "old_default": 5, "restor": 5, "ngrid": 5, "assembl": 5, "string": 5, "accompani": 5, "coordin": 5, "own": 5, "ny": 5, "throughout": 5, "ninc": 5, "intial": 5, "add_training_data": 5, "883": 5, "691": 5, "821": 5, "428": 5, "772": 5, "893": 5, "531": 5, "713": 5, "832": 5, "921": 5, "433": 5, "894": 5, "533": 5, "714": 5, "831": 5, "922": 5, "435": 5, "631": 5, "715": 5, "923": 5, "436": 5, "895": 5, "shrink": 5, "fashion": 5, "tunabl": 5, "definit": 5, "inc": 5, "piec": 5, "wise": 5, "compress": 5, "span": 5, "expand": 5, "impli": 5, "rapid": 5, "th": 5, "design": 5, "leav": 5, "preserv": 5, "postiv": 5, "evolut": 5, "unmodifi": 5, "contigu": 5, "neg": 5, "mainli": 5, "dimenion": 5, "make_uniform": 5, "prealloc": 5, "pair": 5, "view": 5, "plotter": 5, "pyplot": 5, "extract_grid": 5, "1e15": 5, "svdcut": 5, "1e": 5, "distribtuion": 5, "fab": 5, "233": 5, "associ": 5, "affect": 5, "ideal": 5, "adapt_to_pdf": 5, "misbehav": 5, "overflow": 5, "rescal": 5, "eigenvalu": 5, "eig_max": 5, "amelior": 5, "invert": 5, "conserv": 5, "underli": 5, "multipli": 5, "At": 5, "_rescal": 5, "scalar": 5, "buffer": 5, "offset": 5, "stride": 5, "ndarrai": 5, "element": 5, "augment": 5, "v": 5, "k": 5, "except": 5, "valueerror": 5, "repackag": 5, "format": 5, "l": 5, "alia": 5, "reslist": 5, "itg": 5, "ur": 5, "unbias": 5, "0112": 5, "9785": 5, "9980": 5, "0067": 5, "0010": 5, "9996": 5, "0006": 5, "0020": 5, "0051": 5, "0022": 5, "0046": 5, "9976": 5, "instanc": 5, "pdfravg": 5, "pdfravgarrai": 5, "sens": 5, "getattr": 5, "min_neval_batch": [4, 5], "minimum": 5, "move": 5, "disk": 5, "temporari": 5, "h5py": [4, 5], "shoot": 5, "arrasi": 5, "981": 4, "9825": 4, "9915": 4, "9933": 4, "9931": 4}, "objects": {"": [[5, 0, 0, "-", "vegas"]], "vegas": [[5, 1, 1, "", "AdaptiveMap"], [5, 1, 1, "", "BatchIntegrand"], [5, 1, 1, "", "Integrator"], [5, 3, 1, "", "LBatchIntegrand"], [5, 1, 1, "", "PDFIntegrator"], [5, 1, 1, "", "RAvg"], [5, 1, 1, "", "RAvgArray"], [5, 1, 1, "", "RAvgDict"], [5, 1, 1, "", "RBatchIntegrand"], [5, 4, 1, "", "batchintegrand"], [5, 4, 1, "", "lbatchintegrand"], [5, 4, 1, "", "ravg"], [5, 4, 1, "", "rbatchintegrand"]], "vegas.AdaptiveMap": [[5, 2, 1, "", "__call__"], [5, 2, 1, "", "adapt"], [5, 2, 1, "", "adapt_to_samples"], [5, 2, 1, "", "add_training_data"], [5, 2, 1, "", "clear"], [5, 3, 1, "", "dim"], [5, 2, 1, "", "extract_grid"], [5, 3, 1, "", "grid"], [5, 3, 1, "", "inc"], [5, 2, 1, "", "invmap"], [5, 2, 1, "", "jac"], [5, 2, 1, "", "jac1d"], [5, 2, 1, "", "make_uniform"], [5, 2, 1, "", "map"], [5, 3, 1, "", "ninc"], [5, 2, 1, "", "settings"], [5, 2, 1, "", "show_grid"]], "vegas.Integrator": [[5, 2, 1, "", "__call__"], [5, 2, 1, "", "random"], [5, 2, 1, "", "random_batch"], [5, 2, 1, "", "set"], [5, 2, 1, "", "settings"]], "vegas.PDFIntegrator": [[5, 2, 1, "", "__call__"]], "vegas.RAvg": [[5, 3, 1, "", "Q"], [5, 2, 1, "", "add"], [5, 3, 1, "", "chi2"], [5, 3, 1, "", "dof"], [5, 2, 1, "", "extend"], [5, 3, 1, "", "itn_results"], [5, 3, 1, "", "mean"], [5, 3, 1, "", "sdev"], [5, 3, 1, "", "sum_neval"], [5, 2, 1, "", "summary"]], "vegas.RAvgArray": [[5, 3, 1, "", "Q"], [5, 2, 1, "", "add"], [5, 3, 1, "", "chi2"], [5, 3, 1, "", "dof"], [5, 2, 1, "", "extend"], [5, 3, 1, "", "itn_results"], [5, 3, 1, "", "sum_neval"], [5, 2, 1, "", "summary"]], "vegas.RAvgDict": [[5, 3, 1, "", "Q"], [5, 2, 1, "", "add"], [5, 3, 1, "", "chi2"], [5, 3, 1, "", "dof"], [5, 2, 1, "", "extend"], [5, 3, 1, "", "itn_results"], [5, 3, 1, "", "sum_neval"], [5, 2, 1, "", "summary"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:attribute", "4": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "function", "Python function"]}, "titleterms": {"how": 0, "vega": [0, 2, 4, 5], "work": 0, "import": 0, "sampl": 0, "The": [0, 3], "map": [0, 4], "adapt": 0, "stratifi": 0, "integrand": [1, 4], "c": 1, "fortran": 1, "cffi": 1, "cython": 1, "f2py": 1, "document": 2, "indic": 2, "tabl": 2, "case": 3, "studi": 3, "bayesian": [3, 4], "curv": 3, "fit": 3, "problem": 3, "A": 3, "solut": 3, "variat": 3, "tutori": 4, "introduct": [4, 5], "basic": 4, "integr": [4, 5], "multipl": 4, "simultan": 4, "calcul": 4, "distribut": 4, "faster": 4, "processor": 4, "sum": 4, "save": 4, "result": 4, "automat": 4, "precondit": 4, "stratif": 4, "jacobian": 4, "random": 4, "number": 4, "gener": 4, "implement": 4, "note": 4, "modul": 5, "object": 5, "adaptivemap": 5, "pdfintegr": 5, "other": 5, "function": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})