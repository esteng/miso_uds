class Entity:

    # Sometimes there are different ways to say the same thing.
    entity_map = {
        'Netherlands': ['Dutch'],
        'Shichuan': ['Sichuan'],
        'France': ['Frence'],
        'al-Qaida': ['Al-Qaida'],
        'Gorazde': ['Gerlaridy'],
        'Sun': ['Solar'],
        'China': ['Sino'],
        'America': ['US', 'U.S.'],
        'U.S.': ['US'],
        'Georgia': ['GA'],
        'Pennsylvania': ['PA', 'PA.'],
        'Missouri': ['MO', 'MO.'],
        'WWII': ['WW2'],
        'WWI': ['WW1'],
        'Iran': ['Ian'],
        'Jew': ['Semitism', 'Semites'],
        'Islam': ['Muslim'],
        'influenza': ['flu'],
    }

    unknown_entity_type = 'ENTITY'

    def __init__(self, span=None, node=None, ner_type=None, amr_type=None, confidence=0):
        self.span = span
        self.node = node
        self.ner_type = ner_type
        self.amr_type = amr_type
        self.confidence = confidence