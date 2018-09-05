import os

def main():
    print ('Starting pre-processing ...')
    os.system('python pre_process.py')
    os.system('python levenshtein.py')
    os.system('python cwva.py')
    os.system('python rnn.py -t true -c config.json')
    os.system('python rnn.py -t false -c config.json')
    os.system('python evaluate.py')

if __name__ == '__main__':
    main()