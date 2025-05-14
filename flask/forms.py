# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, IntegerField, FloatField
from wtforms.validators import DataRequired, NumberRange

class CarInsuranceAssessmentForm(FlaskForm):
    # Client Info
    # These names should align with the keys in form_data_dict in data_processing.py
    # and ultimately with the column names used in training if they are categorical
    sexe = SelectField('Gender (SX)', validators=[DataRequired()]) # Maps to SX
    age_client = IntegerField('Client Age (AGE)', validators=[DataRequired(), NumberRange(min=18, message="Client age must be 18 or over.")]) # Maps to AGE
    civilite = SelectField('Civil Status (CIV)', validators=[DataRequired()]) # Maps to CIV
    delegation = SelectField('Main Residence (Delegation) (DLG)', validators=[DataRequired()]) # Maps to DLG
    activite = SelectField('Client Activity / Profession (ACT)', validators=[DataRequired()]) # Maps to ACT

    # Vehicle Info
    marque = SelectField('Vehicle Brand (MRQ)', validators=[DataRequired()]) # Maps to MRQ
    carrosserie = SelectField('Vehicle Body Type (CRS)', validators=[DataRequired()]) # Maps to CRS
    usage = SelectField('Vehicle Usage Type (USG)', validators=[DataRequired()]) # Maps to USG
    # 'classe' in your form seems to be 'CLS' in training (numeric)
    classe = SelectField('Risk Class Assignment (CLS)', validators=[DataRequired()]) # Maps to CLS
    energie = SelectField('Fuel Type (EN)', validators=[DataRequired()]) # Maps to EN

    # Technical Details that map to NUMERIC_FEATURES in training
    # Ensure these names match the keys expected by data_processing.py
    anciennete = IntegerField('Client Seniority (Years with Insurer) (ANC)', validators=[DataRequired(), NumberRange(min=0)]) # Maps to ANC
    age_objet_assuree = IntegerField('Vehicle Age (Years) (AGO)', validators=[DataRequired(), NumberRange(min=0, max=90)]) # Maps to AGO
    puissance = IntegerField('Vehicle Horsepower (Fiscal) (PSS)', validators=[DataRequired(), NumberRange(min=0)]) # Maps to PSS
    place = IntegerField('Number of Seats in Vehicle (PLA)', validators=[DataRequired(), NumberRange(min=1)]) # Maps to PLA
    charge_utile = FloatField('Payload Capacity (Tons, if applicable) (CU)', validators=[DataRequired(), NumberRange(min=0.0)]) # Maps to CU

    # Financial Info that map to NUMERIC_FEATURES
    # These are not explicitly in your NUMERIC_FEATURES list in the training script (VV, VN)
    # but your training map.py has "valeur_ranges".
    # Let's assume valeur_venale -> VV and valeur_neuve -> VN
    # If these are not used by the model, they can be removed or just used for display.
    # If they ARE used, they need to be added to NUMERIC_FEATURES in training and here.
    # For now, I'll assume they are needed as VV and VN.
    valeur_venale = IntegerField('Vehicle Current Market Value (DT) (VV)', validators=[DataRequired(), NumberRange(min=0)])
    valeur_neuve = IntegerField('Vehicle Original Price (DT) (VN)', validators=[DataRequired(), NumberRange(min=0)])

    submit = SubmitField('Assess Car Insurance Risk')

    def __init__(self, mappings_dict_for_form_choices, *args, **kwargs):
        super(CarInsuranceAssessmentForm, self).__init__(*args, **kwargs)
        # Populate choices dynamically using the provided mappings
        # These keys MUST match the CATEGORICAL_FEATURES names from training script (or their mapping key in map.py)
        self.sexe.choices = [(k, k) for k in mappings_dict_for_form_choices["sx"].keys()] # "sexe" in form, "sx" in map.py
        self.civilite.choices = [(k, k) for k in mappings_dict_for_form_choices["civ"].keys()] # "civilite" in form, "civ" in map.py
        self.delegation.choices = [(k, k) for k in mappings_dict_for_form_choices["dlg"].keys()] # "delegation" in form, "dlg" in map.py
        self.activite.choices = [(k, k) for k in mappings_dict_for_form_choices["act"].keys()] # "activite" in form, "act" in map.py
        self.marque.choices = [(k, k) for k in mappings_dict_for_form_choices["mrq"].keys()] # "marque" in form, "mrq" in map.py
        self.carrosserie.choices = [(k, k) for k in mappings_dict_for_form_choices["crs"].keys()] # "carrosserie" in form, "crs" in map.py
        self.usage.choices = [(k, k) for k in mappings_dict_for_form_choices["usg"].keys()] # "usage" in form, "usg" in map.py
        self.energie.choices = [(k, k) for k in mappings_dict_for_form_choices["en"].keys()] # "energie" in form, "en" in map.py

        # For 'classe' (CLS), if it's numeric like in training, the choices should be the numbers themselves.
        # Assuming 'classe' in mappings_dict_for_form_choices is a list/set of numeric-like strings
        # e.g., {"0.0", "1.0", ...}
        # The training script's `map.py` has this under "classe" key.
        classe_values = mappings_dict_for_form_choices.get("classe", {"0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "11.0"})
        self.classe.choices = sorted([(str(val), str(val)) for val in classe_values], key=lambda x: float(x[0]))