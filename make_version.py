import re 
import sys 

try:
    outputfile = sys.argv[1] 
except:
    print('*** no output file specified')
    exit(1)

# find version number from setup.cfg
pattern = re.compile('^\w*version[ =]*(.*)$')
with open('setup.cfg', 'r') as ifile:
    for line in ifile.readlines():
        m = pattern.match(line[:-1])
        if m is not None:
            version = f'{m.group(1)}'
            break
    else:
        print('failed to find version number')
        exit(1)

# create version file
with open(outputfile, 'w') as ofile:
    ofile.write(f"__version__ = '{version}'\n")
    print(f'created {outputfile} with __version__ = {version}')
    
