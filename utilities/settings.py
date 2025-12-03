from getpass import getuser
from pathlib import Path
# the user_name should be the name of your session
user_name = getuser()
paths = {'remi': {'basepath': Path('/home/remi/TDS/Programmation/Support/Busquets/'),
                  'dlcpath': Path('/home/remi/TDS/Programmation/Support/Busquets/AstroFear/data/dlc'),
                  'fppath': Path('/home/remi/TDS/Programmation/Support/Busquets/AstroFear/data/photometry'),
                  'table_path': Path('/home/remi/TDS/Programmation/Support/Busquets/AstroFear/data/Mice.txt'),
                  'figures': Path('/home/remi/TDS/Programmation/Support/Busquets/AstroFear/figures'),
                  'succinate': Path('/home/remi/TDS/Programmation/Support/Busquets/Data/Serotonin_Succinate'),
                  'cfc': Path('/home/remi/TDS/Programmation/Support/Busquets/Data/CFC'),
                  'poly': Path('/home/remi/TDS/Programmation/Support/Busquets/Data/CFC/dat'),
                  'astros': Path('/mnt/Data_1/NAS/Data/Busquet/Astros'),
                  'datapath': Path('/mnt/Data_1/NAS/Data/Busquets/ASTROS_fused'),
                  # 'spc_neurons': Path('/home/remi/Aquineuro/Data/busquets/SPC_neurons'),
                  'spc_neurons': Path('/mnt/Data_1/NAS/Data/Busquets/SPC_neurons'),
                  'J60_PC':  Path('/mnt/Data_1/NAS/Data/Busquets/J60_during_PC'),
                  # 'astros': Path('/home/remi/Aquineuro/Data/busquets/'),
                  'astro_fig': Path('/mnt/Data_1/NAS/Data/Busquet/Astros/figures'),
                  'chemo': Path('/mnt/Data_1/NAS/Data/Busquets/Chemogenetics_experiment'),
                  'chemo_elife': Path('/mnt/Data_1/NAS/Data/Busquets/Chemogenetics_eLife')},
         'cramon': {'basepath': Path('C:/Users/cramon/Desktop/Python'),
                     'dlcpath': Path('Z:/Lab Projects/Mineco ASTROCAD/1. Fiber photometry (GCaMP in ASTROS) - Carla + Júlia/5. Fused Data (3rd batch+ACEA ttment)/ASTROS_fused'), 
                     'fppath': Path('Z:/Lab Projects/Mineco ASTROCAD/1. Fiber photometry (GCaMP in ASTROS) - Carla + Júlia/5. Fused Data (3rd batch+ACEA ttment)/ASTROS_fused'),
                     # 'datapath': Path('Z:/Lab Projects/Mineco ASTROCAD/1. Fiber photometry (GCaMP in ASTROS) - Carla + Júlia/5. Fused Data (3rd batch+ACEA ttment)/ASTROS_fused'), #rel_tags.csv
                     'datapath': Path(r'Z:\Lab Projects\ERCstG_HighMemory\Data\Julia\10. eLife revisions\3_FiberPhotometry_CamKII_GiDREADDs_CRD\Gi+GCaMP Mechanism'), #analysis general
                     # 'datapath': Path(r'Z:\Lab Projects\Mineco ASTROCAD\3. Gi DREADDs ASTROS - Carla\2. Gi+GCaMP mechanism\Analysis_filtered\Figures'), #boxplot_transients
                     'table_path': Path('Z:/Lab Projects/Mineco ASTROCAD/1. Fiber photometry (GCaMP in ASTROS) - Carla + Júlia/5. Fused Data (3rd batch+ACEA ttment)/ASTROS_fused/Mice.txt'),
                     'figures': Path('C:/Users/cramon/Desktop/Python/Figures')},
         'ashley': {'spc_neurons': Path(r'C:\Users\ashley.Ashley_Dell\SynologyDrive2\SPC_neurons')}
         }

upaths = paths[user_name]

sites_names={1: 'dHipp', 2: 'vHipp'}


