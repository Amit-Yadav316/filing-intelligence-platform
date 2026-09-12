"""Map an Item number to a canonical section label.

Rule-based on purpose. The Item numbering in 10-K and 10-Q filings is fixed by
regulation, so the mapping is a lookup, not a classification problem. Training a
classifier for this would be a model with no error budget and no benefit over a
table that is right by construction - the sort of thing that looks like machine
learning and is actually a worse dictionary.

The fallback classifier TASKS.md permits is deliberately not built: the rules
resolve every Item in the corpus, so there is nothing for it to do.
"""

from __future__ import annotations

from enum import StrEnum


class Section(StrEnum):
    BUSINESS = "Business"
    RISK_FACTORS = "Risk Factors"
    UNRESOLVED_STAFF_COMMENTS = "Unresolved Staff Comments"
    CYBERSECURITY = "Cybersecurity"
    PROPERTIES = "Properties"
    LEGAL_PROCEEDINGS = "Legal Proceedings"
    MINE_SAFETY = "Mine Safety Disclosures"
    MARKET_FOR_EQUITY = "Market for Registrant's Common Equity"
    SELECTED_FINANCIAL_DATA = "Selected Financial Data"
    MDA = "Management's Discussion and Analysis"
    MARKET_RISK = "Quantitative and Qualitative Disclosures About Market Risk"
    FINANCIAL_STATEMENTS = "Financial Statements"
    ACCOUNTANT_CHANGES = "Changes in and Disagreements with Accountants"
    CONTROLS = "Controls and Procedures"
    OTHER_INFORMATION = "Other Information"
    FOREIGN_JURISDICTIONS = "Foreign Jurisdictions Preventing Inspections"
    DIRECTORS = "Directors and Corporate Governance"
    EXECUTIVE_COMPENSATION = "Executive Compensation"
    SECURITY_OWNERSHIP = "Security Ownership"
    RELATED_TRANSACTIONS = "Certain Relationships and Related Transactions"
    ACCOUNTANT_FEES = "Principal Accountant Fees and Services"
    EXHIBITS = "Exhibits and Financial Statement Schedules"
    FORM_SUMMARY = "Form 10-K Summary"
    QUARTERLY_FINANCIALS = "Financial Statements (unaudited)"
    QUARTERLY_MDA = "Management's Discussion and Analysis"
    QUARTERLY_MARKET_RISK = "Quantitative and Qualitative Disclosures About Market Risk"
    QUARTERLY_CONTROLS = "Controls and Procedures"
    UNKNOWN = "Unknown"


# 10-K item numbering.
_ANNUAL: dict[str, Section] = {
    "1": Section.BUSINESS,
    "1A": Section.RISK_FACTORS,
    "1B": Section.UNRESOLVED_STAFF_COMMENTS,
    "1C": Section.CYBERSECURITY,
    "2": Section.PROPERTIES,
    "3": Section.LEGAL_PROCEEDINGS,
    "4": Section.MINE_SAFETY,
    "5": Section.MARKET_FOR_EQUITY,
    "6": Section.SELECTED_FINANCIAL_DATA,
    "7": Section.MDA,
    "7A": Section.MARKET_RISK,
    "8": Section.FINANCIAL_STATEMENTS,
    "9": Section.ACCOUNTANT_CHANGES,
    "9A": Section.CONTROLS,
    "9B": Section.OTHER_INFORMATION,
    "9C": Section.FOREIGN_JURISDICTIONS,
    "10": Section.DIRECTORS,
    "11": Section.EXECUTIVE_COMPENSATION,
    "12": Section.SECURITY_OWNERSHIP,
    "13": Section.RELATED_TRANSACTIONS,
    "14": Section.ACCOUNTANT_FEES,
    "15": Section.EXHIBITS,
    "16": Section.FORM_SUMMARY,
}

# 10-Q reuses the low Item numbers for entirely different content: Item 1 is
# the financial statements, not the business description. Keying both forms off
# one table would mislabel every quarterly filing.
_QUARTERLY: dict[str, Section] = {
    "1": Section.QUARTERLY_FINANCIALS,
    "2": Section.QUARTERLY_MDA,
    "3": Section.QUARTERLY_MARKET_RISK,
    "4": Section.QUARTERLY_CONTROLS,
    # Part II of a 10-Q.
    "1A": Section.RISK_FACTORS,
    "5": Section.OTHER_INFORMATION,
    "6": Section.EXHIBITS,
}

# Sections an analyst actually asks about; used to weight retrieval and to
# report accuracy by section.
PRIMARY_SECTIONS = frozenset(
    {
        Section.BUSINESS,
        Section.RISK_FACTORS,
        Section.MDA,
        Section.FINANCIAL_STATEMENTS,
        Section.LEGAL_PROCEEDINGS,
        Section.CONTROLS,
        Section.QUARTERLY_FINANCIALS,
        Section.QUARTERLY_MDA,
    }
)


class SectionTagger:
    """Item number plus form type to a canonical section label."""

    @staticmethod
    def normalise_item(item: str) -> str:
        return item.strip().upper().removeprefix("ITEM").strip(" .:")

    def tag(self, item_number: str | None, form: str = "10-K") -> Section:
        if not item_number:
            return Section.UNKNOWN
        key = self.normalise_item(item_number)
        table = _QUARTERLY if form.upper().startswith("10-Q") else _ANNUAL
        return table.get(key, Section.UNKNOWN)

    def is_primary(self, section: Section) -> bool:
        return section in PRIMARY_SECTIONS
