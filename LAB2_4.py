import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('sales_data.csv')
tot_prof = data['total_profit']
month = data['month_number']
plot1 = plt.figure(1)
plt.plot(month, tot_prof, color='r', marker='o', markerfacecolor='k', linestyle='-', linewidth=1.5)
plt.xlabel('Month')
plt.ylabel('Total profit')
plt.grid()
plt.title('TOTAL PROFIT')

facecream = data['facecream']
facewash = data['facewash']
toothpaste = data['toothpaste']
bathingsoap = data['bathingsoap']
shampoo = data['shampoo']
moisturizer = data['moisturizer']
plot2 = plt.figure(2)
plt.plot(month, facecream, month, facewash, month, toothpaste, month, bathingsoap, month, shampoo, month, moisturizer)
plt.legend(['Face Cream', 'Face Wash', 'Toothpaste', 'Bathing Soap', 'Shampoo', 'Moisturizer'])
plt.grid()

plot3 = plt.figure(3)
plt.scatter(month, toothpaste)
plt.grid()

plt4 = plt.figure(4)
plt.bar(month, bathingsoap)
plt.grid()
# plt.savefig('Bathingsoap.png')

plt5 = plt.figure(5)
plt.hist(tot_prof, bins=7)
plt.grid()
plt.show()
