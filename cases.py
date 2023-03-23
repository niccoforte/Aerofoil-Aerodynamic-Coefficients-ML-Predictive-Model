from bs4 import BeautifulSoup
import re
import os
import urllib.request as urllib2
import pandas as pd


def get_AFT_cases(directory='dat/case-dat'):
    """ """

    baseFlpth = "http://airfoiltools.com"

    html_all = urllib2.urlopen("http://airfoiltools.com/search/airfoils").read()
    soup_all = BeautifulSoup(html_all, 'html.parser')

    links_all = [link['href'] for link in soup_all.find_all('a', href=re.compile("/airfoil/details"))]

    linknames = [link[25:-3].lower() for link in links_all]
    filenames_dir = [str(file)[11:].lower() for file in os.scandir(directory)]
    linknames_new = []
    for name in linknames:
        count = 0
        for filename in filenames_dir:
            if name in filename:
                count += 1
            else:
                continue
        if count > 3:
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
            if link_Re.endswith('-n5'):
                continue
            elif link_Re.endswith('-n1'):
                continue

            html_Re = urllib2.urlopen(baseFlpth + link_Re).read()
            soup_Re = BeautifulSoup(html_Re, 'html.parser')

            link_csv = soup_Re.find_all('a', href=re.compile("polar/csv"))[0]['href']

            name = link_csv[20:] + '.csv'
            fullfilename = os.path.join(directory, name)
            urllib2.urlretrieve(baseFlpth + link_csv, fullfilename)

            indx += 1

    print(f'-Done. {indx} files copied from http://airfoiltools.com/search/airfoils and saved to ~/{directory}.')


def get_RENNES_cases(directory='dat/rennes-dat/case-dat'):
    baseFlpth = "https://perso.univ-rennes1.fr/laurent.blanchard/Profils/"

    html_all = urllib2.urlopen(baseFlpth).read()
    soup_all = BeautifulSoup(html_all, 'html.parser')

    links_all = [link['href'] for link in soup_all.find_all('a', href=re.compile("/index"))]
    links_all.remove('centrepoussee/index.html')
    links_all.remove('clouet/index.html')

    linknames = [link[:-11].lower() for link in links_all]
    filenames_dir = [str(file)[11:].split('-')[0].lower() for file in os.scandir(directory)]
    linknames_new = []
    for name in linknames:
        count = 0
        for filename in filenames_dir:
            if name in filename:
                count += 1
            else:
                continue
        if count > 5:
            continue
        else:
            linknames_new.append(name)
    links_new = []
    for new in linknames_new:
        for link in links_all:
            if new in link:
                links_new.append(link)

    print('Staring RENNES Case download...')
    indx = 0
    for link in links_new:
        try:
            html_foil = urllib2.urlopen(baseFlpth + link).read()
            soup_foil = BeautifulSoup(html_foil, 'html.parser')

            links_Re = [link_Re['href'] for link_Re in soup_foil.find_all('a', href=re.compile(".txt"))]

            for link_Re in links_Re:
                name = link[:-11] + '-' + link_Re
                fullfilename = os.path.join(directory, name)
                urllib2.urlretrieve(baseFlpth + link[:-10] + link_Re, fullfilename)

                indx += 1

        except Exception as e:
            pass  # print(e)

    print(f'-Done. {indx} files copied from {baseFlpth} and saved to ~/{directory}.')


def read_CSVcase(file):
    """ """

    top = pd.read_csv(file, nrows=8)
    bottom = pd.read_csv(file, skiprows=9)

    name = top.iloc[1, 0][:-3].lower().replace('.','').replace('_','-')
    Re = float(top.iloc[2, 0])

    alphas = []
    Cls = []
    Cds = []
    for indx, row in bottom.iterrows():
        alpha = float(row.Alpha)
        Cl = float(row.Cl)
        Cd = float(row.Cd)

        alphas.append(alpha)
        Cls.append(Cl)
        Cds.append(Cd)

    return name, Re, alphas, Cls, Cds


def read_TXTcase(file):
    with open(file, "r") as f:
        lines = f.readlines()
        dat = []
        for line in lines:
            line = line.strip()
            dat.append(line)

    name = str(file)[11:-6].split('-')[0].replace('_', '-')
    Re = float(dat[8][28:33]) * 10 ** 6
    dat = dat[12:]

    alphas = []
    Cls = []
    Cds = []
    for row in dat:
        row = row.split('  ')
        alpha = float(row[0])
        Cl = float(row[1])
        Cd = float(row[2])

        alphas.append(alpha)
        Cls.append(Cl)
        Cds.append(Cd)

    return name, Re, alphas, Cls, Cds


def create_cases(directory='dat/case-dat', ext='csv'):
    """ """

    if directory == 'dat/case-dat':
        print('Creating Cases DataFrame...')
    elif directory == 'dat/rennes-dat/case-dat':
        print('Creating Rennes Cases DataFrame...')

    cases_df = pd.DataFrame(columns=['file', 'Re', 'alpha', 'Cl', 'Cd'])
    for file in os.scandir(directory):
        if file.name.endswith('.' + ext):
            try:
                if ext == 'csv':
                    name, Re, alphas, Cls, Cds = read_CSVcase(file)
                elif ext == 'txt':
                    name, Re, alphas, Cls, Cds = read_TXTcase(file)

                name = [name.lower()] * len(alphas)
                Re = [Re] * len(alphas)

                case_df = pd.DataFrame({'file': name, 'Re': Re, 'alpha': alphas, 'Cl': Cls, 'Cd': Cds})
                cases_df = pd.concat([cases_df, case_df], ignore_index=True)

            except Exception as e:
                pass  # print(e)

    print(f'-Done. DataFrame created successfully with {len(cases_df)} cases.')
    return cases_df


def save_cases(df, file):
    print(f'Saving DataFrame to {file}...')
    df.to_csv(file)
    print('-Done.')


def df_from_csv(file):
    print(f'Extracting Cases DataFrame from {file}...')
    df = pd.read_csv(file, index_col=0)

    print(f'-Done. DataFrame extracted successfully with {len(df)} cases.')
    return df
