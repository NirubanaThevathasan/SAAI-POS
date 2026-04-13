import React from 'react';
import SimpleCrud from './SimpleCrud';

const ExpenseCategories = () => {
    return <SimpleCrud
        title="Expense Categories"
        table="expense_categories"
        fields={[
            { key: 'code', label: 'Code' },
            { key: 'name', label: 'Category Name' }
        ]}
    />;
};

export default ExpenseCategories;
