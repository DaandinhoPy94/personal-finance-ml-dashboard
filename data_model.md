# Financial Data Model

## Core Transaction Data
- **Date**: Wanneer de uitgave plaatsvond
- **Amount**: Hoeveel geld (negatief voor uitgaven, positief voor inkomsten)
- **Description**: Beschrijving van de transactie
- **Category**: Hoofdcategorie (Food, Transport, Entertainment, etc.)
- **Subcategory**: Detailcategorie (Groceries, Gas, Movies, etc.)
- **Payment_Method**: Hoe betaald (Card, Cash, Bank Transfer)
- **Location**: Waar (optional)
- **Tags**: Vrije labels voor extra classificatie

## Derived Fields (berekend door de app)
- **Month/Year**: Voor aggregaties
- **Day_of_Week**: Voor patroon analyse
- **Is_Weekend**: Boolean voor weekend uitgaven
- **Running_Balance**: Lopend saldo
- **Category_Budget_Remaining**: Hoeveel budget over in categorie

## Sample Data Structure
```csv
date,amount,description,category,subcategory,payment_method,location,tags
2024-01-15,-12.50,Albert Heijn groceries,Food,Groceries,Card,Amsterdam,weekly_shopping
2024-01-15,-3.20,Coffee at work,Food,Coffee,Card,Office,
2024-01-16,-45.00,Dinner at restaurant,Food,Dining_Out,Card,Amsterdam,date_night
