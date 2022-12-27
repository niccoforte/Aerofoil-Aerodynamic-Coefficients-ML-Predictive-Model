from bs4 import BeautifulSoup
import re
import os
import urllib.request as urllib2
import pandas as pd


def get_AFT_cases(directory='case_dat'):
    baseFlpth = "http://airfoiltools.com"

    html_all = urllib2.urlopen("http://airfoiltools.com/search/airfoils").read()
    soup_all = BeautifulSoup(html_all, 'html.parser')

    links_all = [link['href'] for link in
                 soup_all.find_all('a', href=re.compile("/airfoil/details"))]  # len(links_all) = 1638

    linknames = [link[25:-3].lower() for link in links_all]
    filenames_dir = [str(file)[11:].rsplit('-')[0].lower() for file in os.scandir(directory)]
    linknames_new = []
    for name in linknames:
        count = 0
        for filename in filenames_dir:
            if filename == name:
                count += 1
        if count == 5:
            continue
        else:
            linknames_new.append(name)
    links_new = []
    for new in linknames_new:
        for link in links_all:
            if new in link:
                links_new.append(link)

    print('Staring AFT Case download...')
    indx = 0
    for link in links_new:
        html_foil = urllib2.urlopen(baseFlpth + link).read()
        soup_foil = BeautifulSoup(html_foil, 'html.parser')

        links_Re = [link_Re['href'] for link_Re in soup_foil.find_all('a', href=re.compile("/polar/details"))]

        for link_Re in links_Re:
            if '-n5' in link_Re:
                continue

            html_Re = urllib2.urlopen(baseFlpth + link_Re).read()
            soup_Re = BeautifulSoup(html_Re, 'html.parser')

            link_csv = soup_Re.find_all('a', href=re.compile("/polar/csv"))[0]['href']
            name = link_csv[20:] + '.csv'

            fullfilename = os.path.join(directory, name)
            urllib2.urlretrieve(baseFlpth + link_csv, fullfilename)

        indx += 1

    print(f' Done. {indx} files copied from http://airfoiltools.com/search/airfoils and saved to ~/{directory}.')


def read_case(file):
    top = pd.read_csv(file, nrows=8)
    bottom = pd.read_csv(file, skiprows=9)

    name = top.iloc[1, 0][:-3]
    Re = top.iloc[2, 0]

    alphas = []
    Cls = []
    Cds = []
    for indx, row in bottom.iterrows():
        alpha = row.Alpha
        Cl = row.Cl
        Cd = row.Cd

        alphas.append(alpha)
        Cls.append(Cl)
        Cds.append(Cd)

    return name, Re, alphas, Cls, Cds


def create_cases(directory='case_dat', ext='csv'):
    """ """

    print('Creating Cases DataFrame...')

    cases_df = pd.DataFrame(columns=['file', 'Re', 'alpha', 'Cl', 'Cd'])
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            name, Re, alphas, Cls, Cds = read_case(file)
            name = [name] * len(alphas)
            Re = [Re] * len(alphas)

            case_df = pd.DataFrame({'file': name, 'Re': Re, 'alpha': alphas, 'Cl': Cls, 'Cd': Cds})
            cases_df = pd.concat([cases_df, case_df], ignore_index=True)

    print(' Cases DataFrame ceated successfully.')

    return cases_df
