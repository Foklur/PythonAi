import pandas as pd
import os


data = {
    'OrderID': ['1001', '1002', '1003', '1004', '1005', '1006', '1007' ,'1008' ,'1009' ,'1010'],
    'Customer': ['Alice', 'Bob', 'Alice', 'Mikel', 'Stive', 'Marii', 'David', 'Anna', 'George', 'Fina'],
    'Prouct':['Laptop', 'Chair', 'Mouse', 'Table', 'Lamp', 'Keyboard', 'Phone','Bed','Wi-Fi','TW'],
    'Category': ['Electronics', 'Furniture', 'Electronics', 'Furniture', 'Electronics', 'Electronics', 'Electronics','Furniture', 'Electronics', 'Electronics'],
    'Quantity': [1, 2, 3, 1, 4, 1, 2, 1, 2, 1],
    'Price': [1500, 180, 25, 300, 50, 100, 600, 500, 30, 750],
    'OrderDate': ['2023-06-01', '2023-06-03', '2023-06-05', '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09','2025-05-17','2025-02-26','2024-11-24',]
}

df = pd.DataFrame(data)

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

df['TotalAmount'] = df['Quantity'] * df['Price']

total= df['TotalAmount'].sum()

average_total = df['TotalAmount'].mean()

orders_per_customer = df['Customer'].value_counts()

large_orders = df[df['TotalAmount'] > 500]

sorted_df = df.sort_values(by='OrderDate', ascending=False)

filter_orders = df[(df['OrderDate'] >= '2023-06-05')& (df['OrderDate'] <= '2023-06-10')]

top_total_customers = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3) 

print("DataFrame:")

print("Сумарний дохід магазину:", total)
print("Середнє значення TotalAmount:", average_total)
print("Кількість замовлень по кожному клієнту:\n", orders_per_customer)
print("\nЗамовлення що > 500:\n", large_orders)
print("\nВідсортована таблиця за OrderDate :\n", sorted_df)
print("\nЗамовлення з 5 по 10 червня включно:\n", filter_orders)
print("\nТоп 3 клієнтів за загальною сумою покупок:\n", top_total_customers)