import csv
import random
import os

random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("modules", exist_ok=True)

regions = ["North", "South", "East", "West", "Central"]
customer_types = ["Consumer", "Corporate", "Small Business", "Home Office"]
categories = {
    "Electronics": [("Smartphone", 699.99), ("Laptop", 1199.50), ("Headphones", 149.00), ("Smartwatch", 249.99)],
    "Furniture": [("Desk Chair", 189.00), ("Executive Desk", 450.00), ("Bookshelf", 120.50), ("Desk Lamp", 45.00)],
    "Office Supplies": [("Binder (Pack of 5)", 18.50), ("Paper Ream", 9.99), ("Gel Pen Set", 12.00), ("Stapler Heavy-Duty", 24.50)],
    "Technology": [("Wireless Mouse", 29.99), ("Mechanical Keyboard", 89.99), ("USB-C Docking Station", 79.50), ("External SSD 1TB", 109.99)]
}

names = [
    "Aarav Sharma", "Priya Patel", "Rahul Verma", "Sneha Reddy", "Vikram Singh",
    "Ananya Iyer", "Rohan Mehta", "Kavya Nair", "Aditya Joshi", "Pooja Rao",
    "Manish Gupta", "Divya Deshmukh", "Siddharth Das", "Neha Kulkarni", "Karan Malhotra",
    "Ishaan Saxena", "Meera Sen", "Arjun Pillai", "Tanvi Choudhury", "Nikhil Bhat"
]

rows = []
start_date_ts = 1672531200 # 2023-01-01
day_sec = 86400

for i in range(1, 201):
    order_id = f"ORD-2023-{1000 + i}"
    day_offset = random.randint(1, 360)
    import datetime
    date_val = (datetime.date(2023, 1, 1) + datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
    
    cust_name = random.choice(names)
    region = random.choice(regions)
    cust_type = random.choice(customer_types)
    cat = random.choice(list(categories.keys()))
    prod, unit_price = random.choice(categories[cat])
    
    # Quantity
    if i in [25, 110]:  # Outlier quantities
        quantity = random.randint(45, 60)
    else:
        quantity = random.randint(1, 12)
        
    discount = round(random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25]), 2)
    
    # Calculate Sales & Profit
    sales = round(quantity * unit_price * (1.0 - discount), 2)
    if i == 50:  # Outlier sales
        sales = 14500.00
    
    base_cost = unit_price * 0.60
    profit = round(sales - (quantity * base_cost), 2)
    
    # Intentional missing values for testing
    if i in [12, 45, 88]:
        cust_name = ""
    if i in [18, 95]:
        region = ""
    if i in [33, 142]:
        discount = ""
    if i in [67, 155, 189]:
        profit = ""
    if i == 120:
        quantity = ""
        
    rows.append([order_id, date_val, cust_name, region, cust_type, cat, prod, quantity, unit_price, sales, discount, profit])

# Add 4 duplicate rows intentionally
rows.append(rows[5].copy())
rows.append(rows[20].copy())
rows.append(rows[75].copy())
rows.append(rows[130].copy())

with open("data/sample_sales_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Order_ID", "Order_Date", "Customer_Name", "Region", "Customer_Type", "Category", "Product", "Quantity", "Unit_Price", "Sales", "Discount", "Profit"])
    writer.writerows(rows)

print(f"Generated sample_sales_data.csv with {len(rows)} records (including duplicates and missing values).")
