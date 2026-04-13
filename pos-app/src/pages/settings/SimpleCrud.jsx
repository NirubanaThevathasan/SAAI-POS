import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaTrash, FaPlus } from 'react-icons/fa';

const SimpleCrud = ({ title, table, fields }) => {
    const items = useLiveQuery(() => db[table].toArray());
    const [formData, setFormData] = useState(fields.reduce((acc, field) => ({ ...acc, [field.key]: '' }), {}));

    const handleAdd = async (e) => {
        e.preventDefault();
        await db[table].add(formData);
        setFormData(fields.reduce((acc, field) => ({ ...acc, [field.key]: '' }), {}));
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete this item?')) {
            await db[table].delete(id);
        }
    };

    return (
        <div>
            <h3>{title}</h3>
            <div className="card mb-3">
                <h4>Add New</h4>
                <form onSubmit={handleAdd} style={{ display: 'flex', gap: '10px', alignItems: 'flex-end' }}>
                    {fields.map(field => (
                        <div key={field.key} style={{ flex: 1 }}>
                            <label>{field.label}</label>
                            <input
                                type="text"
                                value={formData[field.key]}
                                onChange={e => setFormData({ ...formData, [field.key]: e.target.value })}
                                required
                                style={{ width: '100%', padding: '8px' }}
                            />
                        </div>
                    ))}
                    <button type="submit" className="btn btn-primary"><FaPlus /> Add</button>
                </form>
            </div>

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left' }}>
                            {fields.map(field => <th key={field.key} style={{ padding: '10px' }}>{field.label}</th>)}
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {items?.map(item => (
                            <tr key={item.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                {fields.map(field => <td key={field.key} style={{ padding: '10px' }}>{item[field.key]}</td>)}
                                <td style={{ padding: '10px' }}>
                                    <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(item.id)}><FaTrash /></button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default SimpleCrud;
