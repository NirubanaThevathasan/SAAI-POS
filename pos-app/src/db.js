import Dexie from 'dexie';

export const db = new Dexie('POSDatabase');

db.version(5).stores({
    // People
    users: '++id, username, password, email, group_id',
    user_groups: '++id, name',
    customers: '++id, name, phone, email, city, address, customer_group_id, credit_limit, discount, tax_number',
    suppliers: '++id, name, phone, email, city, address, tax_number, payment_terms',
    billers: '++id, name, phone, email, city',

    // Settings & Configuration
    settings: 'key',
    warehouses: '++id, code, name',
    tax_rates: '++id, name, code, rate',
    currencies: '++id, code, name, symbol, exchange_rate',
    customer_groups: '++id, name, percentage',
    printers: '++id, name, type, connection_type, ip_address, port',
    email_templates: '++id, name, subject, body',
    categories: '++id, code, name',
    brands: '++id, code, name',
    units: '++id, code, name',
    expense_categories: '++id, code, name',
    payment_methods: '++id, name',

    // Inventory
    products: '++id, name, code, category_id, brand_id, unit_id, type',
    variants: '++id, product_id, name, code',
    batches: '++id, product_id, variant_id, batch_no, expiry_date',
    stock: '++id, [product_id+variant_id+warehouse_id], product_id, warehouse_id',

    // Transactions
    sales: '++id, date, reference_no, customer_id, status, payment_status, warehouse_id',
    sale_items: '++id, sale_id, product_id, variant_id',
    sale_payments: '++id, sale_id, date, method, amount',

    purchases: '++id, date, reference_no, supplier_id, status, payment_status, warehouse_id',
    quotations: '++id, date, reference_no, customer_id, status, warehouse_id',
    transfers: '++id, date, reference_no, from_warehouse_id, to_warehouse_id, status',
    returns: '++id, date, reference_no, type, transaction_id, customer_id, supplier_id, status',
    expenses: '++id, date, reference, category_id, warehouse_id, amount',

    adjustments: '++id, date, reference_no, warehouse_id, type',
    stock_counts: '++id, date, reference_no, warehouse_id, type',

    registers: '++id, open_time, close_time, status, user_id',
    suspended_sales: '++id, date, customer_id',

    deliveries: '++id, sale_id, date, status, customer_id, reference_no',

    calendar_events: '++id, title, start, end',
    notifications: '++id, type, message, read, date, related_id',

    // Module: Customer Suggestions
    product_requests: '++id, item_name, status, date',

    // Module: Shop Notes / General
    shop_notes: '++id, title, type, priority, date',
});

// Pre-populate settings if empty
db.on('populate', () => {
    // Default User Groups
    db.user_groups.bulkAdd([
        {
            id: 1, name: 'Admin', permissions: {
                // Products
                products_view: true, products_add: true, products_edit: true, products_delete: true,
                // Sales
                sales_view: true, sales_add: true, sales_edit: true, sales_delete: true,
                // Purchases
                purchases_view: true, purchases_add: true, purchases_edit: true, purchases_delete: true,
                // Expenses
                expenses_view: true, expenses_add: true, expenses_edit: true, expenses_delete: true,
                // Customers
                customers_view: true, customers_add: true, customers_edit: true, customers_delete: true,
                // Suppliers
                suppliers_view: true, suppliers_add: true, suppliers_edit: true, suppliers_delete: true,
                // Transfers
                transfers_view: true, transfers_add: true, transfers_edit: true, transfers_delete: true,
                // Quotations
                quotations_view: true, quotations_add: true, quotations_edit: true, quotations_delete: true,
                // Returns
                returns_view: true, returns_add: true, returns_edit: true, returns_delete: true,
                // Users
                users_view: true, users_add: true, users_edit: true, users_delete: true,
                // POS
                pos_access: true, pos_discount: true, pos_price_edit: true, pos_hold_sale: true,
                pos_returns: true, pos_manual_item: true, pos_clear_cart: true,
                // Admin
                view_reports: true, manage_settings: true
            }
        },
        {
            id: 2, name: 'Staff', permissions: {
                products_view: true, products_add: true, products_edit: true,
                sales_view: true, sales_add: true,
                customers_view: true, customers_add: true,
                quotations_view: true, quotations_add: true,
                pos_access: true, pos_hold_sale: true, pos_manual_item: true
            }
        },
        {
            id: 3, name: 'Cashier', permissions: {
                products_view: true,
                sales_view: true,
                pos_access: true
            }
        }
    ]);

    db.settings.bulkAdd([
        { key: 'site_name', value: 'SAAI POS' },
        { key: 'currency_code', value: 'LKR' },
        { key: 'default_tax_rate', value: 0 },
        { key: 'default_warehouse', value: 1 }
    ]);
    db.tax_rates.add({ name: 'No Tax', code: 'VAT', rate: 0, type: 1 });
    db.categories.add({ code: 'GEN', name: 'General' });
    db.warehouses.add({ code: 'WH01', name: 'Main Warehouse', address: 'Unknown' });
    db.units.bulkAdd([{ code: 'pc', name: 'Piece' }, { code: 'kg', name: 'Kilogram' }, { code: 'box', name: 'Box' }]);
    db.payment_methods.bulkAdd([{ name: 'Cash' }, { name: 'Card' }, { name: 'UPI' }]);
    db.currencies.bulkAdd([
        { code: 'LKR', name: 'Sri Lankan Rupee', symbol: 'Rs.', exchange_rate: 1 },
        { code: 'INR', name: 'Indian Rupee', symbol: '₹', exchange_rate: 0.28 },
        { code: 'USD', name: 'US Dollar', symbol: '$', exchange_rate: 0.0033 },
        { code: 'EUR', name: 'Euro', symbol: '€', exchange_rate: 0.0030 }
    ]);

    // Default Admins
    db.users.bulkAdd([
        { id: 1, username: 'admin', password: '123', email: 'admin@pos.com', group_id: 1 },
        { id: 2, username: 'admin_main', password: 'admin123', email: 'main_admin@pos.com', group_id: 1 }
    ]);
});

export const resetDatabase = async (options = {}) => {
    const { keepStorage } = options;

    const inventoryTables = [
        'products', 'variants', 'batches', 'stock',
        'categories', 'brands', 'units', 'warehouses'
    ];

    const tables = db.tables.map(table => table.name);

    for (const table of tables) {
        if (keepStorage && inventoryTables.includes(table)) {
            continue;
        }
        await db[table].clear();
    }

    window.location.reload();
};
