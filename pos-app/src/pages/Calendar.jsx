import React, { useState, useEffect } from 'react';
import { db } from '../db';
import { useLiveQuery } from 'dexie-react-hooks';
import { FaChevronLeft, FaChevronRight, FaPlus, FaTrash } from 'react-icons/fa';

const Calendar = () => {
    const [currentDate, setCurrentDate] = useState(new Date());
    const [selectedDate, setSelectedDate] = useState(new Date());
    const [showModal, setShowModal] = useState(false);
    const [newEvent, setNewEvent] = useState({ title: '', start: '', end: '', description: '' });

    // Fetch events for the current month view (or all for simplicity in small scale)
    const events = useLiveQuery(() => db.calendar_events.toArray());

    const getDaysInMonth = (date) => {
        const year = date.getFullYear();
        const month = date.getMonth();
        const days = new Date(year, month + 1, 0).getDate();
        const firstDay = new Date(year, month, 1).getDay(); // 0 = Sunday
        return { days, firstDay };
    };

    const monthData = getDaysInMonth(currentDate);

    const handlePrevMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    };

    const handleNextMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    };

    const handleDayClick = (day) => {
        const date = new Date(currentDate.getFullYear(), currentDate.getMonth(), day);
        setSelectedDate(date);
        // Pre-fill modal with clicked date
        const dateStr = date.toISOString().split('T')[0];
        setNewEvent({ ...newEvent, start: dateStr, end: dateStr });
        setShowModal(true);
    };

    const handleSaveEvent = async () => {
        if (!newEvent.title || !newEvent.start) return;
        await db.calendar_events.add({
            title: newEvent.title,
            start: newEvent.start,
            end: newEvent.end || newEvent.start,
            description: newEvent.description
        });
        setShowModal(false);
        setNewEvent({ title: '', start: '', end: '', description: '' });
    };

    const handleDeleteEvent = async (id, e) => {
        e.stopPropagation();
        if (window.confirm('Delete this event?')) {
            await db.calendar_events.delete(id);
        }
    };

    const renderCalendarGrid = () => {
        const slots = [];
        // Empty slots for days before the 1st
        for (let i = 0; i < monthData.firstDay; i++) {
            slots.push(<div key={`empty-${i}`} style={{ background: '#f9f9f9', borderRight: '1px solid #eee', borderBottom: '1px solid #eee' }}></div>);
        }

        // Days
        for (let d = 1; d <= monthData.days; d++) {
            const dateStr = new Date(currentDate.getFullYear(), currentDate.getMonth(), d).toISOString().split('T')[0];
            const dayEvents = events?.filter(e => {
                const start = e.start.split('T')[0];
                return start === dateStr;
            }) || [];

            const isToday = new Date().toISOString().split('T')[0] === dateStr;

            slots.push(
                <div
                    key={d}
                    onClick={() => handleDayClick(d)}
                    style={{
                        minHeight: '100px',
                        padding: '5px',
                        borderRight: '1px solid #eee',
                        borderBottom: '1px solid #eee',
                        background: isToday ? '#e3f2fd' : 'white',
                        cursor: 'pointer',
                        position: 'relative'
                    }}
                    className="calendar-day"
                >
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{d}</div>
                    <div className="events-list">
                        {dayEvents.map(ev => (
                            <div
                                key={ev.id}
                                style={{
                                    background: '#3498db',
                                    color: 'white',
                                    padding: '2px 4px',
                                    borderRadius: '3px',
                                    fontSize: '11px',
                                    marginBottom: '2px',
                                    overflow: 'hidden',
                                    whiteSpace: 'nowrap',
                                    textOverflow: 'ellipsis',
                                    display: 'flex',
                                    justifyContent: 'space-between'
                                }}
                                title={ev.title}
                            >
                                <span>{ev.title}</span>
                                <span onClick={(e) => handleDeleteEvent(ev.id, e)} style={{ cursor: 'pointer', marginLeft: '5px' }}>&times;</span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }
        return slots;
    };

    return (
        <div className="calendar-page" style={{ height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
            <div className="calendar-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <h2 style={{ margin: 0 }}>
                        {currentDate.toLocaleString('default', { month: 'long' })} {currentDate.getFullYear()}
                    </h2>
                    <div style={{ display: 'flex', gap: '5px' }}>
                        <button className="btn btn-light btn-sm" onClick={handlePrevMonth}><FaChevronLeft /></button>
                        <button className="btn btn-light btn-sm" onClick={() => setCurrentDate(new Date())}>Today</button>
                        <button className="btn btn-light btn-sm" onClick={handleNextMonth}><FaChevronRight /></button>
                    </div>
                </div>
                <button className="btn btn-primary" onClick={() => setShowModal(true)}><FaPlus /> Add Event</button>
            </div>

            <div className="calendar-grid" style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gridAutoRows: 'minmax(100px, 1fr)', borderTop: '1px solid #eee', borderLeft: '1px solid #eee' }}>
                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                    <div key={day} style={{ padding: '10px', textAlign: 'center', fontWeight: 'bold', background: '#f8f9fa', borderBottom: '1px solid #eee', borderRight: '1px solid #eee' }}>
                        {day}
                    </div>
                ))}

                {renderCalendarGrid()}
            </div>

            {showModal && (
                <div className="modal-overlay">
                    <div className="modal-content" style={{ width: '400px' }}>
                        <h3>Add Event / Reminder</h3>
                        <div className="form-group">
                            <label>Title</label>
                            <input
                                type="text"
                                className="form-control"
                                value={newEvent.title}
                                onChange={e => setNewEvent({ ...newEvent, title: e.target.value })}
                                autoFocus
                            />
                        </div>
                        <div className="form-group">
                            <label>Date</label>
                            <input
                                type="date"
                                className="form-control"
                                value={newEvent.start}
                                onChange={e => setNewEvent({ ...newEvent, start: e.target.value })}
                            />
                        </div>
                        <div className="form-group">
                            <label>Description (Optional)</label>
                            <textarea
                                className="form-control"
                                rows="3"
                                value={newEvent.description}
                                onChange={e => setNewEvent({ ...newEvent, description: e.target.value })}
                            />
                        </div>
                        <div className="modal-actions">
                            <button className="btn btn-secondary" onClick={() => setShowModal(false)}>Cancel</button>
                            <button className="btn btn-primary" onClick={handleSaveEvent}>Save Event</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Calendar;
