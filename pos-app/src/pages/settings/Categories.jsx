import React from 'react';
import SimpleCrud from './SimpleCrud';

const Categories = () => {
    return <SimpleCrud
        title="Categories"
        table="categories"
        fields={[
            { key: 'code', label: 'Code' },
            { key: 'name', label: 'Category Name' }
        ]}
    />;
};

export default Categories;
