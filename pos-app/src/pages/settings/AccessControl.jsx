import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaPlus, FaTrash, FaEdit, FaSave, FaTimes, FaUsers } from 'react-icons/fa';
import { useAuth } from '../../context/AuthContext';

const permissionGroups = [
    {
        name: 'General Modules',
        items: [
            { module: 'products', label: 'Products', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'sales', label: 'Sales', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'purchases', label: 'Purchases', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'expenses', label: 'Expenses', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'customers', label: 'Customers', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'suppliers', label: 'Suppliers', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'transfers', label: 'Transfers', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'quotations', label: 'Quotations', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'returns', label: 'Returns', actions: ['view', 'add', 'edit', 'delete'] },
            { module: 'users', label: 'Users/Staff', actions: ['view', 'add', 'edit', 'delete'] },
        ]
    },
    {
        name: 'POS Terminal',
        items: [
            { key: 'pos_access', label: 'Access Terminal', description: 'Can enter POS screen' },
            { key: 'pos_discount', label: 'Apply Discount', description: 'Can add discounts/tax' },
            { key: 'pos_price_edit', label: 'Edit Price', description: 'Can change unit prices' },
            { key: 'pos_hold_sale', label: 'Hold Sale', description: 'Can suspend/resume sales' },
            { key: 'pos_manual_item', label: 'Manual Items', description: 'Can add custom items' },
            { key: 'pos_clear_cart', label: 'Clear Cart', description: 'Can purge sale list' },
        ]
    },
    {
        name: 'Administrative',
        items: [
            { key: 'view_reports', label: 'Reports Viewer', description: 'Access to analytics' },
            { key: 'manage_settings', label: 'System Settings', description: 'Technical configuration' },
        ]
    }
];

const AccessControl = () => {
    const groups = useLiveQuery(() => db.user_groups.toArray());
    const { refreshPermissions } = useAuth();
    const [selectedGroup, setSelectedGroup] = useState(null);
    const [isAddingMode, setIsAddingMode] = useState(false);
    const [isEditingName, setIsEditingName] = useState(false);
    const [newName, setNewName] = useState('');

    const handleTogglePermission = async (permKey) => {
        if (!selectedGroup) return;

        const groupToUpdate = await db.user_groups.get(selectedGroup.id);
        const newPermissions = {
            ...groupToUpdate.permissions,
            [permKey]: !groupToUpdate.permissions?.[permKey]
        };

        await db.user_groups.update(selectedGroup.id, { permissions: newPermissions });

        // Update current session if the modified group is the one the user belongs to
        await refreshPermissions();

        // Update selectedGroup to reflect changes in UI matrix
        setSelectedGroup({ ...groupToUpdate, permissions: newPermissions });
    };

    const handleAddGroup = async (e) => {
        e.preventDefault();
        if (!newName.trim()) return;

        const id = await db.user_groups.add({
            name: newName,
            permissions: {}
        });

        const addedGroup = await db.user_groups.get(id);
        setSelectedGroup(addedGroup);
        setIsAddingMode(false);
        setNewName('');
    };

    const handleDeleteGroup = async (id, e) => {
        e.stopPropagation();
        if (selectedGroup?.id === id && (selectedGroup.name === 'Admin' || selectedGroup.id === 1)) {
            return alert('Cannot delete the Admin group.');
        }

        if (window.confirm('Are you sure you want to delete this group? Users assigned to this group may lose access.')) {
            await db.user_groups.delete(id);
            if (selectedGroup?.id === id) setSelectedGroup(null);
        }
    };

    const handleSaveName = async () => {
        if (!newName.trim() || !selectedGroup) return;
        if (selectedGroup.name === 'Admin') return alert('Cannot rename Admin group.');

        await db.user_groups.update(selectedGroup.id, { name: newName });
        setSelectedGroup({ ...selectedGroup, name: newName });
        setIsEditingName(false);
    };

    return (
        <div className="access-control">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div>
                    <h3 style={{ margin: 0 }}>User Groups & Permissions</h3>
                    <p className="text-muted" style={{ margin: 0 }}>Manage access levels for your staff.</p>
                </div>
                <button
                    className="btn btn-primary"
                    onClick={() => { setIsAddingMode(true); setIsEditingName(false); setNewName(''); }}
                    disabled={isAddingMode}
                >
                    <FaPlus /> New Group
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: '30px' }}>
                <div className="group-list">
                    {isAddingMode && (
                        <form onSubmit={handleAddGroup} style={{ marginBottom: '15px', padding: '15px', border: '1px solid #0ea5e9', borderRadius: '8px', background: '#f0f9ff' }}>
                            <input
                                autoFocus
                                type="text"
                                className="form-control mb-2"
                                placeholder="Group Name..."
                                value={newName}
                                onChange={e => setNewName(e.target.value)}
                                style={{ width: '100%', padding: '8px' }}
                            />
                            <div style={{ display: 'flex', gap: '5px' }}>
                                <button type="submit" className="btn btn-primary btn-sm" style={{ flex: 1 }}>Add</button>
                                <button type="button" className="btn btn-light btn-sm" onClick={() => setIsAddingMode(false)} style={{ flex: 1 }}>Cancel</button>
                            </div>
                        </form>
                    )}

                    {groups?.map(group => (
                        <div
                            key={group.id}
                            onClick={() => { setSelectedGroup(group); setIsEditingName(false); setIsAddingMode(false); }}
                            style={{
                                padding: '12px 15px',
                                border: '1px solid #ddd',
                                borderRadius: '8px',
                                marginBottom: '8px',
                                cursor: 'pointer',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                background: selectedGroup?.id === group.id ? '#f0f9ff' : '#fff',
                                borderColor: selectedGroup?.id === group.id ? '#0ea5e9' : '#ddd',
                                transition: 'all 0.2s'
                            }}
                        >
                            <span style={{ fontWeight: selectedGroup?.id === group.id ? '700' : '500' }}>{group.name}</span>
                            {group.name !== 'Admin' && group.id !== 1 && (
                                <button
                                    className="btn-icon text-danger"
                                    onClick={(e) => handleDeleteGroup(group.id, e)}
                                    title="Delete Group"
                                    style={{ padding: '4px', opacity: 0.6 }}
                                >
                                    <FaTrash size={12} />
                                </button>
                            )}
                        </div>
                    ))}
                </div>

                <div className="permissions-matrix card" style={{ padding: '0', overflow: 'hidden' }}>
                    {!selectedGroup ? (
                        <div style={{ textAlign: 'center', padding: '100px 50px', color: '#666' }}>
                            <FaUsers size={40} style={{ opacity: 0.2, marginBottom: '15px' }} />
                            <div>Select a group from the left to manage permissions.</div>
                        </div>
                    ) : (
                        <div>
                            <div style={{ padding: '20px', borderBottom: '1px solid #eee', background: '#fcfcfc', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    {isEditingName ? (
                                        <div style={{ display: 'flex', gap: '5px' }}>
                                            <input
                                                autoFocus
                                                type="text"
                                                className="form-control"
                                                value={newName}
                                                onChange={e => setNewName(e.target.value)}
                                                style={{ padding: '4px 8px', fontSize: '1.2rem', fontWeight: '700', width: '200px' }}
                                            />
                                            <button className="btn btn-success btn-sm" onClick={handleSaveName}><FaSave /></button>
                                            <button className="btn btn-light btn-sm" onClick={() => setIsEditingName(false)}><FaTimes /></button>
                                        </div>
                                    ) : (
                                        <>
                                            <h4 style={{ margin: 0 }}>{selectedGroup.name}</h4>
                                            {selectedGroup.name !== 'Admin' && selectedGroup.id !== 1 && (
                                                <button
                                                    className="btn-link"
                                                    onClick={() => { setIsEditingName(true); setNewName(selectedGroup.name); }}
                                                    style={{ fontSize: '0.8rem', padding: 0, color: '#0ea5e9', background: 'none', border: 'none', cursor: 'pointer' }}
                                                >
                                                    <FaEdit /> Rename
                                                </button>
                                            )}
                                        </>
                                    )}
                                </div>
                                {selectedGroup.name === 'Admin' && (
                                    <span className="badge badge-warning" style={{ background: '#fef3c7', color: '#92400e', padding: '5px 10px' }}>
                                        Full Access (Admin)
                                    </span>
                                )}
                            </div>

                            <div style={{ padding: '20px', maxHeight: '70vh', overflowY: 'auto' }}>
                                {permissionGroups.map((group, gIdx) => (
                                    <div key={gIdx} style={{ marginBottom: '30px' }}>
                                        <h5 style={{
                                            background: '#f8fafc',
                                            padding: '8px 15px',
                                            borderRadius: '6px',
                                            margin: '0 0 15px 0',
                                            fontSize: '0.9rem',
                                            color: 'var(--primary-color)',
                                            borderLeft: '4px solid var(--primary-color)'
                                        }}>
                                            {group.name}
                                        </h5>

                                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                            {group.items.map((item, iIdx) => (
                                                <div
                                                    key={iIdx}
                                                    style={{
                                                        display: 'grid',
                                                        gridTemplateColumns: '260px 1fr',
                                                        alignItems: 'center',
                                                        padding: '10px 15px',
                                                        background: iIdx % 2 === 0 ? 'transparent' : '#fcfcfc',
                                                        borderRadius: '4px'
                                                    }}
                                                >
                                                    <div>
                                                        <div style={{ fontWeight: '600', fontSize: '0.9rem' }}>{item.label}</div>
                                                        {item.description && <div style={{ fontSize: '0.75rem', color: '#888' }}>{item.description}</div>}
                                                    </div>

                                                    <div style={{ display: 'flex', gap: '15px 25px', justifyContent: 'flex-start', flexWrap: 'wrap' }}>
                                                        {item.actions ? (
                                                            item.actions.map(action => {
                                                                const permKey = `${item.module}_${action}`;
                                                                const isEnabled = !!selectedGroup.permissions?.[permKey];
                                                                return (
                                                                    <label key={action} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: selectedGroup.name === 'Admin' ? 'default' : 'pointer', fontSize: '0.85rem' }}>
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={selectedGroup.name === 'Admin' || isEnabled}
                                                                            onChange={() => selectedGroup.name !== 'Admin' && handleTogglePermission(permKey)}
                                                                            disabled={selectedGroup.name === 'Admin'}
                                                                        />
                                                                        {action.charAt(0).toUpperCase() + action.slice(1)}
                                                                    </label>
                                                                );
                                                            })
                                                        ) : (
                                                            <div style={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'flex-end' }}>
                                                                <div
                                                                    onClick={() => selectedGroup.name !== 'Admin' && handleTogglePermission(item.key)}
                                                                    style={{
                                                                        width: '40px',
                                                                        height: '20px',
                                                                        borderRadius: '10px',
                                                                        background: (selectedGroup.name === 'Admin' || !!selectedGroup.permissions?.[item.key]) ? '#10b981' : '#ddd',
                                                                        position: 'relative',
                                                                        cursor: selectedGroup.name === 'Admin' ? 'default' : 'pointer',
                                                                        transition: 'background 0.3s',
                                                                        opacity: selectedGroup.name === 'Admin' ? 0.7 : 1
                                                                    }}
                                                                >
                                                                    <div style={{
                                                                        width: '14px',
                                                                        height: '14px',
                                                                        borderRadius: '50%',
                                                                        background: '#fff',
                                                                        position: 'absolute',
                                                                        top: '3px',
                                                                        left: (selectedGroup.name === 'Admin' || !!selectedGroup.permissions?.[item.key]) ? '23px' : '3px',
                                                                        transition: 'left 0.3s'
                                                                    }} />
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AccessControl;
