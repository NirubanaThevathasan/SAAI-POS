import React from 'react';
import SimpleCrud from './SimpleCrud';

const Brands = () => {
    return <SimpleCrud
        title="Brands"
        table="brands"
        fields={[
            { key: 'code', label: 'Code' },
            { key: 'name', label: 'Brand Name' }
        ]}
    />;
};

export default Brands;
