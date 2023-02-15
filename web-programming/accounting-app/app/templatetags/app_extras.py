from django import template

register = template.Library()

@register.simple_tag(takes_context=True)
def journal_table(context, pfl):
    request = context.get('request')    
    trans_table = []
    debit_total, credit_total = 0, 0
    for trans in pfl.transaction_set.all():
        if trans.trans_type == 'dbt':
            trans_table.append((trans.date, trans.trans_name, trans.amount, ''))
            debit_total += trans.amount
        else:
            trans_table.append((trans.date, trans.trans_name, '', trans.amount))
            credit_total += trans.amount
    context = {'tbl': trans_table, 'dt': debit_total, 'ct': credit_total}
    return context