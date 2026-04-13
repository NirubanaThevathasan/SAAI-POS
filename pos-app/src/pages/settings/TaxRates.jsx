import React from 'react';
import SimpleCrud from './SimpleCrud';

const TaxRates = () => {
    return <SimpleCrud
        title="Tax Rates"
        table="tax_rates"
        fields={[
            { key: 'name', label: 'Name' },
            { key: 'rate', label: 'Rate (%)' }
        ]}
    />;
};

export default TaxRates;
