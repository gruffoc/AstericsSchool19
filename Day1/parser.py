__author__ = "Stefano Mandelli"
__date__ = "$08-apr-2019 12:00:00$"
__version__ = "0.1"

from argparse import ArgumentParser, RawTextHelpFormatter


def funzione_do_something():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-n', '--name', default="NOME", type=str, help="Inserire il Nome")
    parser.add_argument('-c', '--cognome', default="COGNOME", type=str, help="Inserire il Cognome")
    parser.add_argument('-e', '--eta', default="ETA", type=str, help="Inserire l'eta`")
    parser.add_argument('-s', '--stronzaggine', action='store_true', help="Set the stronzaggine") # non vuole nulla a fianco


    args = parser.parse_args()

    print(args.name)

    if args.stronzaggine:
        print("Argomento Stronzaggine attivo")
    else:
        print("Argomento Stronzaggine disattivo")



if __name__ == "__main__":
    funzione_do_something()
