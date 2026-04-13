import React from 'react';
import SimpleCrud from './SimpleCrud';

const Warehouses = () => {
    return <SimpleCrud
        title="Warehouses"
        table="warehouses"
        fields={[
            { key: 'code', label: 'Code' },
            { key: 'name', label: 'Name' },
            { key: 'address', label: 'Address' }
        ]}
    />;
};

export default Warehouses;
