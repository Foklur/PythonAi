import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='f(x) = x^2 * sin(x)')
plt.title('Графік функції : f(x) = x^2 * sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

mean = 5
std_dev = 2
data = np.random.normal(mean, std_dev, 1000)
plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Гістограма випадкових чисел (N(5, 2))')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

hobbies = ['Ігри', 'Програмування', 'Музика', 'Читання']
shares = [35, 10, 30, 25] 
plt.figure(figsize=(6, 6))
plt.pie(shares, labels=hobbies, autopct='%1.1f%%', startangle=140)
plt.title('Мої улюблені хобі')
plt.axis('equal')  
plt.show()

fruits = ['Яблуко', 'Банан', 'Апельсин', 'Груша']
data = {fruit: np.random.normal(loc=150, scale=20, size=100) for fruit in fruits}
plt.figure(figsize=(8, 5))
sns.boxplot(data=list(data.values()))
plt.xticks(ticks=range(len(fruits)), labels=fruits)
plt.title('Box-plot маси фруктів')
plt.ylabel('Маса (гр)')
plt.grid(True)
plt.show()
