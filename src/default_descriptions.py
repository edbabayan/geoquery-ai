table_description_list_default = [
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.daily_allocation",
        "table_description": "This table captures the result of allocating daily production and injection volumes to individual wells. It provides production and injection details such as flow direction (production or injection), material disposition codes (natural, gas-lift, ESP), material type (oil, water, gas), production date, flowing hours (duration). It provides detailed insights into production and injection volumes across multiple fields and reservoirs as well."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.inactive_string",
        "table_description": "This table provides inactive wells and strings on a monthly basis. It contains inactive reasons such as inactive category, problem name, string status (inactive, active, abandoned), string health (healthy, problematic, etc.). It also tracks downtime start dates, estimated action dates & expected rates across multiple assets, fields, and reservoirs."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.real_time_corporate_pi",
        "table_description": "This table captures time series real-time operations sensor data for oil production & injection wells. Attributes include pressure, temperature, choke size, valve status, injection, company name, well name, string type, and real-time data tag name for unique well identifiers (UWI)."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_reservoir",
        "table_description": "Maps wells and strings to their associated reservoirs. Contains unique well identifiers, string name (LS, SS, ST/TB), associated field name. Each row represents a unique well (UWI) with its reservoir, field, and string designation."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.string",
        "table_description": "Same as well_reservoir: maps wells and strings to reservoirs. Contains unique well identifiers, string name (LS, SS, ST/TB), field name. Each row represents a unique well (UWI) with reservoir, field, and string designation."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.string_event",
        "table_description": "Captures well and string status (flowing, down, injecting, etc.) and reasons for status. Contains well and string identifiers, reservoir details, event date, reasons, descriptions, remarks, choke size, pressures, temperatures, flow rates, with timestamps for event start/end. Granularity is at string level within wells."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.unified_pressure_test",
        "table_description": "Consolidates pressure surveys/test data for categories like BHCIP, PBU, PFO, GRAD, BHFP for producers/injectors. Includes test date, reservoir pressure (datum, mean, gauge), BHP (bottom hole pressures), permeability, productivity index, reservoir, gauge, and service company details."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well",
        "table_description": "Master data for wells: company, project, field, UWI, well name, type (producer, observer, injector), operator, coordinates, elevation, depth, current/previous status, spud/completion dates, and unique well identifiers."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_completion",
        "table_description": "Represents well completion downhole equipment data. Includes completion type, dimensions (OD, ID, length), installation/removal dates, inner/outer diameters, equipment lengths for tracking completion components within wells."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_log_index",
        "table_description": "Represents logging services on wells. Provides data for services (GR, PLT, RST, etc.), hole conditions, mud properties, formation details, depth intervals, logging dates, service provider, fluid characteristics supporting subsurface analysis and well performance."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.wellbore",
        "table_description": "Detailed wellbore information: unique well identifiers, drilling metrics, geological targets, coordinates, operational details, bore status, borehole name, operator, depth, formation codes, rig details, spud/completion dates across multiple assets, fields, and reservoirs."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_allowable_limits",
        "table_description": "Well-level data on allowable production/injection rates, technical rates updated monthly. Includes allowable rates, technical rates, material types, field, reservoir, well identifiers, and production/injection limits (start/end dates)."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.field",
        "table_description": "Unified view of fields: field codes, names, associated company, and project codes."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.flow_test",
        "table_description": "Captures detailed flow test data for wells: test type, duration, rates (oil, gas, water), choke size, wellhead pressures, temperatures, chemical compositions. Granular at well and tubing string level for reservoir performance and production analysis over time."
    }

]
