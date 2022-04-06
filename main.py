import pandas as pd
import pulp as pl
import requests

df = pd.read_csv('data/input.csv')

# Create Lists By Position
data = df.copy().reset_index()
data.set_index('Name', inplace=True)
players = data.index.tolist()

data_g = df[df['Pos']== 'G'].copy().reset_index()
data_g.set_index('Name', inplace=True)
goalkeepers = data_g.index.tolist()

data_d = df[df['Pos']== 'D'].copy().reset_index()
data_d.set_index('Name', inplace=True)
defenders = data_d.index.tolist()

data_m = df[df['Pos']== 'M'].copy().reset_index()
data_m.set_index('Name', inplace=True)
midfielders = data_m.index.tolist()

data_f = df[df['Pos']== 'F'].copy().reset_index()
data_f.set_index('Name', inplace=True)
forwards = data_f.index.tolist()

# Create Lists By Team
data_ars = df[df['Team'] == 'Arsenal'].copy().reset_index()
data_ars.set_index('Name', inplace=True)
arsenal = data_ars.index.tolist()

data_avl = df[df['Team'] == 'Aston Villa'].copy().reset_index()
data_avl.set_index('Name', inplace=True)
aston_villa = data_avl.index.tolist()

data_bha = df[df['Team'] == 'Brighton'].copy().reset_index()
data_bha.set_index('Name', inplace=True)
brighton = data_bha.index.tolist()

data_bur = df[df['Team'] == 'Burnley'].copy().reset_index()
data_bur.set_index('Name', inplace=True)
burnley = data_bur.index.tolist()

data_bre = df[df['Team'] == 'Brentford'].copy().reset_index()
data_bre.set_index('Name', inplace=True)
brentford = data_bre.index.tolist()

data_che = df[df['Team'] == 'Chelsea'].copy().reset_index()
data_che.set_index('Name', inplace=True)
chelsea = data_che.index.tolist()

data_cpl = df[df['Team'] == 'Crystal Palace'].copy().reset_index()
data_cpl.set_index('Name', inplace=True)
crystal_palace = data_cpl.index.tolist()

data_eve = df[df['Team'] == 'Everton'].copy().reset_index()
data_eve.set_index('Name', inplace=True)
everton = data_eve.index.tolist()

data_lee = df[df['Team'] == 'Leeds'].copy().reset_index()
data_lee.set_index('Name', inplace=True)
leeds = data_lee.index.tolist()

data_lei = df[df['Team'] == 'Leicester'].copy().reset_index()
data_lei.set_index('Name', inplace=True)
leicester = data_lee.index.tolist()

data_liv = df[df['Team'] == 'Liverpool'].copy().reset_index()
data_liv.set_index('Name', inplace=True)
liverpool = data_liv.index.tolist()

data_mci = df[df['Team'] == 'Man City'].copy().reset_index()
data_mci.set_index('Name', inplace=True)
man_city = data_mci.index.tolist()

data_mun = df[df['Team'] == 'Man Utd'].copy().reset_index()
data_mun.set_index('Name', inplace=True)
man_utd = data_mun.index.tolist()

data_new = df[df['Team'] == 'Newcastle'].copy().reset_index()
data_new.set_index('Name', inplace=True)
newcastle = data_new.index.tolist()

data_nor = df[df['Team'] == 'Norwich'].copy().reset_index()
data_nor.set_index('Name', inplace=True)
norwich = data_nor.index.tolist()

data_sou = df[df['Team'] == 'Southampton'].copy().reset_index()
data_sou.set_index('Name', inplace=True)
southampton = data_sou.index.tolist()

data_tot = df[df['Team'] == 'Spurs'].copy().reset_index()
data_tot.set_index('Name', inplace=True)
tottenham = data_tot.index.tolist()

data_wat = df[df['Team'] == 'Watford'].copy().reset_index()
data_wat.set_index('Name', inplace=True)
watford = data_wat.index.tolist()

data_whu = df[df['Team'] == 'West Ham'].copy().reset_index()
data_whu.set_index('Name', inplace=True)
west_ham = data_whu.index.tolist()

data_wol = df[df['Team'] == 'Wolves'].copy().reset_index()
data_wol.set_index('Name', inplace=True)
wolves = data_wol.index.tolist()

next_gw = (df.keys()[5].split('_')[0])
gw_1 = next_gw + 1
gw_2 = next_gw + 2
gw_3 = next_gw + 3
gw_4 = next_gw + 4
gw_5 = next_gw + 5
gw_6 = next_gw + 6
gw_7 = next_gw + 7

# User Set Variables
budget = 101.9

# Model Set up
model = pl.LpProblem('model', pl.LpMaximize)
solver = pl.PULP_CBC_CMD()

# Decision Variables
lineup = pl.LpVariable.dicts('lineup', players, 0, 1, cat='Binary')
squad = pl.LpVariable.dicts('squad', players, 0, 1, cat='Binary')
captain = pl.LpVariable.dicts('captain', players, 0, 1, cat='Binary')
vicecap = pl.LpVariable.dicts('vicecap', players, 0, 1, cat='Binary')


# Objective Variable
model += pl.lpSum([data.loc[p, f'{next_gw}_Pts'] * (0.9 * lineup[p] + 0.01 * squad[p] + captain[p] + 0.1 * vicecap[p]) for p in players])

# Valid Squad Within Budget
model += pl.lpSum(squad[p] for p in players) == 15
model += pl.lpSum(squad[p] * data.loc[p, 'BV'] for p in players) <= budget
model += pl.lpSum(squad[g] for g in goalkeepers) == 2
model += pl.lpSum(squad[d] for d in defenders) == 5
model += pl.lpSum(squad[m] for m in midfielders) == 5
model += pl.lpSum(squad[f] for f in forwards) == 3

# Valid Lineup Formation
model += pl.lpSum(lineup[p] for p in players) == 11
model += pl.lpSum(lineup[g] for g in goalkeepers) == 1
model += pl.lpSum(lineup[d] for d in defenders) >= 3
model += pl.lpSum(lineup[d] for d in defenders) <= 5
model += pl.lpSum(lineup[m] for m in midfielders) >= 2
model += pl.lpSum(lineup[m] for m in midfielders) <= 5
model += pl.lpSum(lineup[f] for f in forwards) >= 1
model += pl.lpSum(lineup[f] for f in forwards) <= 3

# Valid Captain Numbers
model += pl.lpSum(captain[p] for p in players) == 1
model += pl.lpSum(vicecap[p] for p in players) == 1

# Valid Club Maximum
model += pl.lpSum(squad[x] for x in arsenal) <= 3
model += pl.lpSum(squad[x] for x in aston_villa) <= 3
model += pl.lpSum(squad[x] for x in brentford) <= 3
model += pl.lpSum(squad[x] for x in brighton) <= 3
model += pl.lpSum(squad[x] for x in burnley) <= 3
model += pl.lpSum(squad[x] for x in chelsea) <= 3
model += pl.lpSum(squad[x] for x in crystal_palace) <= 3
model += pl.lpSum(squad[x] for x in everton) <= 3
model += pl.lpSum(squad[x] for x in leeds) <= 3
model += pl.lpSum(squad[x] for x in leicester) <= 3
model += pl.lpSum(squad[x] for x in liverpool) <= 3
model += pl.lpSum(squad[x] for x in man_city) <= 3
model += pl.lpSum(squad[x] for x in man_utd) <= 3
model += pl.lpSum(squad[x] for x in newcastle) <= 3
model += pl.lpSum(squad[x] for x in norwich) <= 3
model += pl.lpSum(squad[x] for x in southampton) <= 3
model += pl.lpSum(squad[x] for x in tottenham) <= 3
model += pl.lpSum(squad[x] for x in watford) <= 3
model += pl.lpSum(squad[x] for x in west_ham) <= 3
model += pl.lpSum(squad[x] for x in wolves) <= 3

# Lineup & Cap Within Squad, C/VC Different
for p in players:
    model += lineup[p] <= squad[p]
    model += captain[p] <= lineup[p]
    model += vicecap[p] <= lineup[p]
    model += vicecap[p] + captain[p] <= 1

# Solve Model
result = model.solve(solver)

print("Captain:")
for p in players:
    if captain[p].varValue >= 0.5:
        print(p)

print("VC:")
for p in players:
   if vicecap[p].varValue >= 0.5:
          print(p)

print("Lineup:")
for p in players:
    if lineup[p].varValue >= 0.5:
        print(p)

print("Squad:")
for p in players:
    if squad[p].varValue >= 0.5:
        print(p)