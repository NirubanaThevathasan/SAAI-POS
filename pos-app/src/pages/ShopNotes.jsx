import React, { useState } from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { useAuth } from '../context/AuthContext';
import { FaPlus, FaCheck, FaTimes, FaEdit, FaTrash, FaLightbulb, FaCommentDots, FaExclamationCircle, FaStickyNote } from 'react-icons/fa';

const ShopNotes = () => {
    const { user } = useAuth();
    const [showModal, setShowModal] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [formData, setFormData] = useState({
        title: '',
        description: '',
        type: 'General',
        priority: 'Medium'
    });

    const notes = useLiveQuery(() => db.shop_notes.orderBy('date').reverse().toArray());

    const resetForm = () => {
        setFormData({
            title: '',
            description: '',
            type: 'General',
            priority: 'Medium'
        });
        setEditingId(null);
    };

    const handleEdit = (note) => {
        setFormData({
            title: note.title,
            description: note.description || '',
            type: note.type || 'General',
            priority: note.priority || 'Medium'
        });
        setEditingId(note.id);
        setShowModal(true);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const data = {
            ...formData,
        };

        if (editingId) {
            await db.shop_notes.update(editingId, data);
        } else {
            await db.shop_notes.add({
                ...data,
                date: new Date(),
                staff_name: user?.username || 'Unknown'
            });
        }

        resetForm();
        setShowModal(false);
    };

    const deleteNote = async (id) => {
        if (window.confirm('Delete this note?')) {
            await db.shop_notes.delete(id);
        }
    };

    const getTypeIcon = (type) => {
        switch (type) {
            case 'Idea': return <FaLightbulb style={{ color: '#f59e0b' }} />;
            case 'Complaint': return <FaExclamationCircle style={{ color: '#ef4444' }} />;
            case 'Feedback': return <FaCommentDots style={{ color: '#3b82f6' }} />;
            default: return <FaStickyNote style={{ color: '#666' }} />;
        }
    };

    const getPriorityColor = (p) => {
        switch (p) {
            case 'High': return '#fee2e2'; // Red-ish bg
            case 'Low': return '#f0fdf4'; // Green-ish bg
            default: return '#fff';
        }
    };

    return (
        <div className="page-content">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div>
                    <h2 style={{ margin: 0 }}>Shop Notes & Feedback</h2>
                    <p style={{ margin: '5px 0 0', color: '#666' }}>Important information, ideas, and complaints.</p>
                </div>
                <button className="btn btn-primary" onClick={() => { resetForm(); setShowModal(true); }}>
                    <FaPlus /> Add Note
                </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
                {notes?.map(note => (
                    <div className="card" key={note.id} style={{
                        background: getPriorityColor(note.priority),
                        borderLeft: `4px solid ${note.priority === 'High' ? '#ef4444' : note.priority === 'Low' ? '#10b981' : '#f59e0b'}`,
                        position: 'relative'
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                {getTypeIcon(note.type)}
                                <span style={{ fontWeight: '600', fontSize: '0.8rem', textTransform: 'uppercase', color: '#666' }}>{note.type}</span>
                            </div>
                            <div style={{ display: 'flex', gap: '5px' }}>
                                <button onClick={() => handleEdit(note)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#6366f1' }}><FaEdit /></button>
                                <button onClick={() => deleteNote(note.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444' }}><FaTrash /></button>
                            </div>
                        </div>

                        <h3 style={{ margin: '0 0 10px 0', fontSize: '1.2rem' }}>{note.title}</h3>
                        <p style={{ whiteSpace: 'pre-wrap', color: '#333', marginBottom: '15px' }}>{note.description}</p>

                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#666', borderTop: '1px solid rgba(0,0,0,0.05)', paddingTop: '10px' }}>
                            <span>Author: {note.staff_name}</span>
                            <span>{new Date(note.date).toLocaleDateString()}</span>
                        </div>

                        <div style={{ position: 'absolute', top: '10px', right: '50px', fontSize: '0.7rem', fontWeight: 'bold', padding: '2px 6px', background: 'rgba(0,0,0,0.1)', borderRadius: '4px' }}>
                            {note.priority} Priority
                        </div>
                    </div>
                ))}

                {notes?.length === 0 && (
                    <div style={{ gridColumn: '1/-1', textAlign: 'center', padding: '40px', color: '#666', background: '#f8fafc', borderRadius: '8px' }}>
                        <FaStickyNote style={{ fontSize: '3rem', opacity: 0.2, marginBottom: '10px' }} />
                        <p>No notes found. Keep track of your ideas here!</p>
                    </div>
                )}
            </div>

            {showModal && (
                <div className="modal-overlay" style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
                }}>
                    <div className="modal-content card" style={{ width: '500px', padding: '24px', background: '#fff' }}>
                        <h3 style={{ marginTop: 0 }}>{editingId ? 'Edit Note' : 'Add New Note'}</h3>
                        <form onSubmit={handleSubmit}>
                            <div style={{ marginBottom: '15px' }}>
                                <label className="form-label">Title *</label>
                                <input required type="text" className="form-control" value={formData.title} onChange={e => setFormData({ ...formData, title: e.target.value })} placeholder="Short summary" />
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                                <div>
                                    <label className="form-label">Type</label>
                                    <select className="form-control" value={formData.type} onChange={e => setFormData({ ...formData, type: e.target.value })}>
                                        <option value="General">General Note</option>
                                        <option value="Idea">Shop Improvement Idea</option>
                                        <option value="Feedback">Customer Feedback</option>
                                        <option value="Complaint">Complaint</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="form-label">Priority</label>
                                    <select className="form-control" value={formData.priority} onChange={e => setFormData({ ...formData, priority: e.target.value })}>
                                        <option value="Low">Low</option>
                                        <option value="Medium">Medium</option>
                                        <option value="High">High</option>
                                    </select>
                                </div>
                            </div>

                            <div style={{ marginBottom: '20px' }}>
                                <label className="form-label">Description / Details</label>
                                <textarea className="form-control" rows="5" value={formData.description} onChange={e => setFormData({ ...formData, description: e.target.value })} placeholder="Enter detailed information here..."></textarea>
                            </div>

                            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                                <button type="submit" className="btn btn-primary">Save Note</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ShopNotes;
