from wasteops.guidance import get_guidance

def test_guidance_always_warns_about_local_rules():
    assert get_guidance('plastic')['local_rules_required'] is True
    assert 'vary' in get_guidance('plastic')['disclaimer']

def test_unknown_gets_safe_guidance():
    assert get_guidance('not-a-class')['local_rules_required'] is True
