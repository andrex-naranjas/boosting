#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Code to improve SVM
#authors: A. Ramirez-Morales and J. Salmon-Gamboa

# visualization moduler

#work in progress

#Pivoting
#print(data_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(data_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#plotting
# g = sns.FacetGrid(data_set, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(data_set, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();
# #fig = grid.get_figure()
# grid.savefig("output.png")

# grid = sns.FacetGrid(data_set, row='Fare', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# grid.savefig("male_female.png")
