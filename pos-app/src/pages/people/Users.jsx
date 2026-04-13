import React, { useState } from 'react';
import { db } from '../../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaTrash, FaPlus, FaEdit } from 'react-icons/fa';
import { useAuth } from '../../context/AuthContext';

const Users = () => {
    const { hasPermission } = useAuth();
    const users = useLiveQuery(() => db.users.toArray());
    const groups = useLiveQuery(() => db.user_groups.toArray());
    const [showForm, setShowForm] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        email: '',
        first_name: '',
        last_name: '',
        group_id: '2', // Default to Staff
        status: 'active'
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (editingId) {
                await db.users.update(editingId, formData);
                setEditingId(null);
            } else {
                await db.users.add(formData);
            }
            setFormData({ username: '', password: '', email: '', first_name: '', last_name: '', group_id: '2', status: 'active' });
            setShowForm(false);
        } catch (error) {
            alert('Error saving user');
        }
    };

    const handleEdit = (user) => {
        setFormData({
            username: user.username,
            password: user.password,
            email: user.email,
            first_name: user.first_name,
            last_name: user.last_name,
            group_id: user.group_id,
            status: user.status
        });
        setEditingId(user.id);
        setShowForm(true);
    };

    const handleDelete = async (id) => {
        if (window.confirm('Delete user?')) {
            await db.users.delete(id);
        }
    };

    const cancelForm = () => {
        setFormData({ username: '', password: '', email: '', first_name: '', last_name: '', group_id: '2', status: 'active' });
        setEditingId(null);
        setShowForm(false);
    };

    const getGroupName = (id) => {
        const group = groups?.find(g => g.id === parseInt(id));
        return group ? group.name : 'Unknown';
    };

    return (
        <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2>Staff / Users</h2>
                {hasPermission('users_add') && (
                    <button className="btn btn-primary" onClick={() => { cancelForm(); setShowForm(!showForm); }}>
                        <FaPlus /> Add User
                    </button>
                )}
            </div>

            {showForm && (
                <div className="card mb-3">
                    <h3>{editingId ? 'Edit User' : 'Add New User'}</h3>
                    <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                        <div className="form-group">
                            <label>Username</label>
                            <input type="text" className="form-control" value={formData.username} onChange={e => setFormData({ ...formData, username: e.target.value })} required style={{ width: '100%', padding: '8px' }} />
                        </div>
                        <div className="form-group">
                            <label>Password</label>
                            <input type="password" className="form-control" value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} required style={{ width: '100%', padding: '8px' }} />
                        </div>
                        <div className="form-group">
                            <label>Email</label>
                            <input type="email" className="form-control" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} required style={{ width: '100%', padding: '8px' }} />
                        </div>
                        <div className="form-group">
                            <label>First Name *</label>
                            <input type="text" className="form-control" value={formData.first_name} onChange={e => setFormData({ ...formData, first_name: e.target.value })} required style={{ width: '100%', padding: '8px', marginTop: '5px' }} placeholder="First Name" />
                        </div>
                        <div className="form-group">
                            <label>Last Name *</label>
                            <input type="text" className="form-control" value={formData.last_name} onChange={e => setFormData({ ...formData, last_name: e.target.value })} required style={{ width: '100%', padding: '8px', marginTop: '5px' }} placeholder="Last Name" />
                        </div>
                        <div className="form-group">
                            <label>Role</label>
                            <select className="form-control" value={formData.group_id} onChange={e => setFormData({ ...formData, group_id: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                {groups?.map(g => <option key={g.id} value={g.id}>{g.name}</option>)}
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Status</label>
                            <select className="form-control" value={formData.status} onChange={e => setFormData({ ...formData, status: e.target.value })} style={{ width: '100%', padding: '8px' }}>
                                <option value="active">Active</option>
                                <option value="inactive">Inactive</option>
                            </select>
                        </div>
                        <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                            <button type="submit" className="btn btn-primary">{editingId ? 'Update' : 'Save User'}</button>
                            <button type="button" className="btn" onClick={cancelForm} style={{ marginLeft: '10px' }}>Cancel</button>
                        </div>
                    </form>
                </div>
            )}

            <div className="card">
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #eee', textAlign: 'left', background: '#f8f9fa' }}>
                            <th style={{ padding: '10px' }}>Username</th>
                            <th style={{ padding: '10px' }}>Name</th>
                            <th style={{ padding: '10px' }}>Email</th>
                            <th style={{ padding: '10px' }}>Role</th>
                            <th style={{ padding: '10px' }}>Status</th>
                            <th style={{ padding: '10px' }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {users?.map(user => (
                            <tr key={user.id} style={{ borderBottom: '1px solid #f9f9f9' }}>
                                <td style={{ padding: '10px' }}>{user.username}</td>
                                <td style={{ padding: '10px' }}>{user.first_name} {user.last_name}</td>
                                <td style={{ padding: '10px' }}>{user.email}</td>
                                <td style={{ padding: '10px' }}>
                                    <span className={`badge badge-${user.group_id === '1' ? 'primary' : 'secondary'}`}>
                                        {getGroupName(user.group_id)}
                                    </span>
                                </td>
                                <td style={{ padding: '10px' }}>
                                    {user.status === 'active'
                                        ? <span style={{ color: 'green', fontWeight: 'bold' }}>Active</span>
                                        : <span style={{ color: 'gray' }}>Inactive</span>}
                                </td>
                                <td style={{ padding: '10px' }}>
                                    {hasPermission('users_edit') && (
                                        <button className="btn-icon" style={{ cursor: 'pointer', marginRight: '10px' }} onClick={() => handleEdit(user)}><FaEdit /></button>
                                    )}
                                    {hasPermission('users_delete') && (
                                        <button className="btn-icon" style={{ color: 'red' }} onClick={() => handleDelete(user.id)}><FaTrash /></button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Users;
